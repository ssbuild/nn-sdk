#include "trt_csdk.h"
#include "../dyloader/trt_loader.h"
#include <algorithm>
//#include "pybind11/pybind11.h"
//#include "pybind11/numpy.h"
//#include "pybind11/cast.h"

C_dylink_trt* g_trt = NULL;


//numpy库
/*if (_import_array() < 0) {
	PyErr_Print();
	PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
	return NULL;
}*/

void C_trt_resource::close() {
	if (m_trt_handle) {
		cc_sdk_trt_delete(m_trt_handle);
		m_trt_handle = NULL;
	}
}


int C_trt_resource::OnCreate(int enable_graph){
	g_trt = C_dylink_trt::Instance();

	try
	{
		char so_path[256] = { 0 };
		{
			auto tmp = PyImport_ImportModule("trt_sdk");
			if (tmp == NULL) {
				PyErr_Print();
				log_err("ModuleNotFoundError: No module named trt_sdk\n");
				return -1;
			}
			PyObject* oret = PyModule_GetFilenameObject(tmp);
			Py_DECREF(tmp);
			if (!oret) {
				PyErr_Print();
				log_err("PyModule_GetFilenameObject failed\n");
				return -1;
			}
			snprintf(so_path, 256, "%s", PyUnicode_AsUTF8(oret));
			Py_DECREF(oret);

			char* pos = strrchr(so_path, '/');
			if (pos) {
				*pos = '\0';
			}
		}
		log_debug("trt_sdk: %s\n", so_path);
		if (!g_trt->load(so_path, m_engine_major,m_engine_minor)) {
			log_err("dylink_trt load failed");
			return -1;
		}
	}catch(std::exception & e) {
		PyErr_Print();
		log_err("%s except %s\n",__FUNCTION__,e.what());
	}

	

	std::string sread;
	std::string sread_plain;
	std::string* model_data = NULL;
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
	log_debug("cc_sdk_trt_new ...\n");
	int ret = cc_sdk_trt_new((const char*)model_data->data(), model_data->size(), m_device_id,
		enable_graph, input_shape_all, input_item_size_all, output_shape_all, output_item_size_all, m_trt_handle);
	if (0 != ret) {
		log_err("cc_sdk_trt_new failed");
		return -1;
	}
	return 0;
}



int C_trt_resource::OnProcess(PyObject** result, int stage,int input_num_, PyObject** inputs_){
	/*typedef int (*f_cc_sdk_trt_process)(const CC_SDK_TRT_HANDLE instance, int net_stage, const void** input_buffer_list,
		const int input_num, void*& output_buf_only_read, int& output_buf_size);*/

	int ret = -1;
	auto net = this->m_net_graph[stage];
	if (net.oInput_.size() != input_num_) {
		log_err("%s bad input num, graph num: %lld , input_num: %lld\n", __FUNCTION__, net.oInput_.size(), input_num_);
		return ret;
	}

	this->m_input_buffer_pt_list.resize(input_num_);
	std::string input_data_type;
	int batch_size = 1;
	for (int i = 0; i < input_num_; ++i) {
		auto& olist = inputs_[i];
		auto& data_shape = net.input_[i].shape;
		py::array input_info(py::reinterpret_borrow<py::object>(olist));
		if (i == 0) {
			auto i_shape = input_info.shape();
			batch_size = i_shape[0];
		}
		int flag = input_info.flags();
		if (flag & NPY_ARRAY_C_CONTIGUOUS || flag & NPY_ARRAY_F_CONTIGUOUS) {
			auto i_ndim = input_info.ndim();
			if (i_ndim != data_shape.size()) {
				log_err("bad input data\n");
				return ret;
			}
			if (batch_size > data_shape[0]) {
				log_err("input batch size must lte graph batch size\n");
				return ret;
			}
			this->m_input_buffer_pt_list[i] = input_info.request().ptr;
		}
		else {
			auto np_type = net.input_[i].data_type;
			int element_size = Get_dsize_by_type(np_type);
			int input_size = std::accumulate(data_shape.begin() + 1, data_shape.end(), batch_size * element_size, [](int a, int b) {return a * b; });
			this->m_input_buffer[i].resize(input_size);
			char* data_pt = this->m_input_buffer[i].data();
			GetList_to_buffer(olist, data_pt);//会改变输入指针
			this->m_input_buffer_pt_list[i] = this->m_input_buffer[i].data();
		}
	}
	this->m_output_buf_only_read.resize(net.output_.size());
	this->m_output_buf_size.resize(net.output_.size());
	ret = cc_sdk_trt_process(this->m_trt_handle, stage, this->m_input_buffer_pt_list.data(), input_num_, batch_size, this->m_output_buf_only_read.data(), this->m_output_buf_size.data());
	if (ret != 0) {
		return ret;
	}

	int out_size = net.output_.size();
	*result = PyTuple_New(out_size);
	for (int i = 0; i < out_size; ++i) {
		auto& o = net.output_[i];
		auto& r_shape = this->m_output_shape;
		r_shape = o.shape;
		r_shape[0] = batch_size;
		
		py::array r_arr(py::dtype(o.dtype_short_str), r_shape, this->m_output_buf_only_read[i]);
		PyTuple_SetItem(*result, i, r_arr.ptr());
		r_arr.inc_ref();
		/*auto arr = PyArray_SimpleNewFromData(r_shape.size(), r_shape.data(), o.data_type, this->m_output_buf_only_read[i]);
		PyArray_ENABLEFLAGS((PyArrayObject*)arr, NPY_ARRAY_OWNDATA);
		PyTuple_SetItem(*result, i, arr);*/
	}
	return ret;
}
