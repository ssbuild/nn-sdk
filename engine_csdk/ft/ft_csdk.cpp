#include "ft_csdk.h"
#include "../dyloader/ft_loader.h"
#include <algorithm>
//#include "pybind11/pybind11.h"
//#include "pybind11/numpy.h"
//#include "pybind11/cast.h"

C_dylink_ft* g_instance = NULL;

//ft_create
//numpy库
/*if (_import_array() < 0) {
	PyErr_Print();
	PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
	return NULL;
}*/

void C_ft_resource::close() {
	if (m_handle) {
		ft_destroy(m_handle);
		m_handle = NULL;
	}
}


int C_ft_resource::OnCreate(int k, float threshold, int predict_label,int dump_label){
	m_k = k;
	m_threshold = threshold;
	m_predict_label = predict_label;
	m_dump_label = dump_label;

	g_instance = C_dylink_ft::Instance();
	try
	{
		char so_path[256] = { 0 };
		{
			auto tmp = PyImport_ImportModule("nn_sdk");
			if (tmp == NULL) {
				PyErr_Print();
				log_err("ModuleNotFoundError: No module named nn_sdk\n");
				return -1;
			}
			PyObject* oret = PyModule_GetFilenameObject(tmp);
			Py_DECREF(tmp);
			if (!oret) {
				log_err("PyModule_GetFilenameObject failed\n");
				PyErr_Print();
				return -1;
			}
			snprintf(so_path, 256, "%s", PyUnicode_AsUTF8(oret));
			Py_DECREF(oret);
#ifdef _WIN32
			char* pos = strrchr(so_path, '\\');
#else
			char* pos = strrchr(so_path, '/');
#endif // _WIN32
			if (pos) {
				*pos = '\0';
			}
		}
		log_debug("nn_sdk: %s\n", so_path);
		if (!g_instance->load(so_path, m_engine_major)) {
			log_err("dylink_ft load failed");
			return -1;
		}
		log_debug("nn-dylink_ft load ok\n");
	}
	catch (std::exception& e) {
		PyErr_Print();
		log_err("%s except %s\n", __FUNCTION__, e.what());
	}

	

	std::string sread;
	std::string sread_plain;
	std::string* model_data = NULL;

	log_debug("load model....\n");
	if (read_file(m_model_dir.c_str(), sread) <= 0) {
		return -1;
	}
	if (m_aes_conf.use) {
		if (0 != tk_aes_decode((uint8_t*)sread.c_str(), sread.size(), sread_plain, m_aes_conf.key, m_aes_conf.iv)) {
			log_err("aes decode failed");
			return -1;
		}
		model_data = &sread_plain;
	}
	else {
		model_data = &sread;
	}

	log_debug("config prepare...\n");
	std::vector<std::vector<std::vector<int>>> input_shape_all, output_shape_all;
	std::vector<std::vector<int>>input_item_size_all, output_item_size_all;
	int net_max_input_num = 1;
	for (auto & item : m_net_graph) {
		std::vector<std::vector<int>> input_shape, output_shape;
		std::vector<int>input_item_size,output_item_size;

		auto& input = item.input_;
		input_shape.resize(input.size());
		input_item_size.resize(input.size());

		if (net_max_input_num < input.size()) {
			net_max_input_num = input.size();
		}

		for (int j = 0; j < input.size();++j) {
			auto & info = input[j];
			auto& cur_shape = input_shape[j];
			cur_shape.insert(cur_shape.end(), info.shape.begin(), info.shape.end());
			input_item_size[j] = Get_dsize_by_type(info.data_type);
		}
		auto& output = item.output_;
		output_shape.resize(output.size());
		output_item_size.resize(output.size());
		for (int j = 0; j < output.size(); ++j) {
			auto& info = output[j];
			auto& cur_shape = output_shape[j];
			cur_shape.insert(cur_shape.end(), info.shape.begin(), info.shape.end());
			output_item_size[j] = Get_dsize_by_type(info.data_type);
		}

		input_shape_all.push_back(input_shape);
		input_item_size_all.push_back(input_item_size);

		output_shape_all.push_back(output_shape);
		output_item_size_all.push_back(output_item_size);
	}

	m_input_buffer_pt_list.resize(net_max_input_num);
	m_input_buffer.resize(net_max_input_num);
	log_debug("ft_new ...\n");
	S_ft_param_inout param;
	param.model_dir = model_data->data();
	param.model_size = model_data->size();
	param.is_model_dir_file = false;
	param.dump_label = m_dump_label;
	param.debug = the_config.log_level >= LOG_LEVEL_DEUBG;
	param.labels = (void*)&m_labels;

	//param.model_dir = m_model_dir.data();
	//param.model_size = m_model_dir.size();
	//param.is_model_dir_file = true;
	int ret = ft_new(param, m_handle);
	if (0 != ret) {
		log_err("ft_new failed");
		return -1;
	}
	log_info("vector_dim %d\n", param.vector_dim);
	m_dim = param.vector_dim;
	return 0;
}



int C_ft_resource::OnProcess(PyObject** result, int stage,int input_num_, PyObject** inputs_){
	/*typedef int (*f_cc_sdk_trt_process)(const CC_SDK_TRT_HANDLE instance, int net_stage, const void** input_buffer_list,
		const int input_num, void*& output_buf_only_read, int& output_buf_size);*/

	int ret = -1;
	auto net = this->m_net_graph[stage];
	if (net.oInput_.size() != input_num_ || input_num_ !=1) {
		log_err("%s bad input num, graph num: %lld , input_num: %lld\n", __FUNCTION__, net.oInput_.size(), input_num_);
		return ret;
	}

	this->m_input_buffer_pt_list.resize(input_num_);
	std::string input_data_type;
	int batch_size = 1;
	for (int i = 0; i < input_num_; ++i) {
		auto& olist = inputs_[i];
		py::array input_info(py::reinterpret_borrow<py::object>(olist));
		if (i == 0) {
			auto i_shape = input_info.shape();
			batch_size = i_shape[0];
		}
		this->m_input_buffer[i].resize(batch_size * sizeof(char*));
		char* data_pt = this->m_input_buffer[i].data();
		GetList_to_buffer(olist, data_pt);//会改变输入指针
		this->m_input_buffer_pt_list[i] = this->m_input_buffer[i].data();

		/*printf("batch_size %d\n", batch_size);
		for (int j = 0; j < batch_size;++j) {
			char** tmp = (char**)(this->m_input_buffer[i].data() + j * sizeof(char*));
			printf("str: %s\n", *tmp);
		}*/
		
	}
	this->m_output_buf_only_read.resize(net.output_.size());
	this->m_output_buf_size.resize(net.output_.size());

	if (!m_predict_label) {
		//typedef int (*f_ft_process)(void* handle, char** text_utf8, int n, int batch, void** output_buf_only_read, int* out_buf_size);
		ret = ft_process(this->m_handle, (char**)this->m_input_buffer_pt_list.data(), input_num_, batch_size, this->m_output_buf_only_read.data(), this->m_output_buf_size.data());
		if (ret != 0) {
			return ret;
		}

		int out_size = net.output_.size();
		*result = PyTuple_New(out_size);

		m_output_shape.resize(2);
		for (int i = 0; i < out_size; ++i) {
			auto& o = net.output_[i];
			/*auto& r_shape = this->m_output_shape;
			r_shape = o.shape;
			r_shape[0] = batch_size;*/
		
			m_output_shape[0] = batch_size;
			m_output_shape[1] = this->m_output_buf_size[i] / batch_size / sizeof(float);
			py::array r_arr(py::dtype(o.dtype_short_str), m_output_shape, this->m_output_buf_only_read[i]);

	
			PyTuple_SetItem(*result, i, r_arr.ptr());
			r_arr.inc_ref();
			/*auto arr = PyArray_SimpleNewFromData(r_shape.size(), r_shape.data(), o.data_type, this->m_output_buf_only_read[i]);
			PyArray_ENABLEFLAGS((PyArrayObject*)arr, NPY_ARRAY_OWNDATA);
			PyTuple_SetItem(*result, i, arr);*/
		}
	}else{
		//typedef int  (*f_ft_process_label)(void* handle, char** text_utf8, int n, int batch, void** output_buf_only_read, int k, int threshold)
		ret = ft_process_label(this->m_handle, (char**)this->m_input_buffer_pt_list.data(), input_num_, batch_size, this->m_output_buf_only_read.data(), m_k, m_threshold);
		if (ret != 0) {
			return ret;
		}
		int out_size = net.output_.size();
		*result = PyTuple_New(out_size);

		m_output_shape.resize(2);
		for (int i = 0; i < out_size; ++i) {
			std::vector<std::vector<std::pair<float, int32_t>>>* predict_list = (std::vector<std::vector<std::pair<float, int32_t>>>*)this->m_output_buf_only_read[i];
			py::list lst(predict_list->size());
			for (int j = 0, j_size = predict_list->size(); j < j_size; ++j) {
				auto& predict = predict_list->at(j);
				py::list sub_lst;
				for (int k = 0, k_size = predict.size(); k < k_size; ++k) {
					auto& map_pair = predict[k];
					py::list end_lst;

					end_lst.append(py::float_((float)map_pair.second));
					end_lst.append(py::float_(map_pair.first));
					sub_lst.append(end_lst);
				}
				lst[j] = sub_lst;
			}
			PyTuple_SetItem(*result, i, lst.ptr());
			lst.inc_ref();
		}
	}
	return ret;
}
