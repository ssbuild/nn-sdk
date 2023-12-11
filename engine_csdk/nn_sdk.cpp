#include "tf_v1/tf_csdk_v1.h"
#include "tf_v2/tf_csdk_v2.h"
#include "onnx/onnx_csdk.h"
#include "trt/trt_csdk.h"
#include "nn_sdk.h"

#include "sdk_py.h"
#include "pybind11/embed.h"

//#include "rapidjson/document.h"
//#include "rapidjson/writer.h"
//#include "rapidjson/stringbuffer.h"


void my_PyDict_SetItemString(PyObject* dict,const char*k, PyObject*v) {
	auto key = PyUnicode_FromString(k);
	PyDict_SetItem(dict,key,v);
	Py_DECREF(key);
}


int sdk_init_cc() {
	try {
		if (!Py_IsInitialized()) {
			Py_Initialize();
		}
		log_info("support data type:\n");
		log_info("INT32: %d\n", NPY_INT);
		log_info("UINT32: %d\n", NPY_UINT);
		log_info("LONG: %d\n", NPY_LONG);
		log_info("ULONG: %d\n", NPY_ULONG);
		log_info("INT64: %d\n", NPY_LONGLONG);
		log_info("UINT64: %d\n", NPY_ULONGLONG);
		log_info("FLOAT: %d\n", NPY_FLOAT);
		log_info("DOUBLE: %d\n", NPY_DOUBLE);
		
		//import_array();
		/*if (_import_array() < 0) { 
			PyErr_Print(); 
			PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import"); 
			return -1;
		}*/
		log_info("%s sucess\n", __FUNCTION__);
		return 0;
	}
	catch (std::exception & e) {
		log_fatal("%s %s\n",__FUNCTION__,e.what());
	}
	return -1;
}

int sdk_uninit_cc() {
	Py_Finalize();
	return 0;
}

SDK_HANDLE_CC sdk_new_cc(const char* json_string) {
	try {
		py::dict dict = py::eval(json_string);
		py::print(dict);
		py::object oret = ns_sdk_py::sdk_new(dict);
		long long handle = py::cast<long long>(oret);
		return handle;
	}
	catch (std::exception& e) {
		log_err("%s %s",__FUNCTION__,e.what());
	}
	return -1;
}



int sdk_delete_cc(SDK_HANDLE_CC handle) {
	int code = -1;
	if (handle) {
		auto engine = ((C_engine_base*)handle);
		delete engine;
	}
	return code;
}



//input_buffer_list N 个一维数组, final_result N个一维数组
int sdk_process_cc(SDK_HANDLE_CC handle, int net_stage, int batch_size, void** input_buffer_list, void** final_result) {

	int code = -1;
	if (!handle) return code;
	auto engine = ((C_engine_base*)handle);
	if (net_stage >= engine->m_net_graph.size()) {
		return code;
	}

	auto graph = engine->m_net_graph[net_stage];
	int input_num = graph.input_.size();


	PyObject* pyresult = NULL;
	bool has_error = false;

	auto& oinput_list = engine->m_lts_input;
	oinput_list.resize(input_num);
	for (int i = 0; i < input_num; ++i) {
		auto & node = graph.input_[i];

		auto tmp_shape = node.shape;
		tmp_shape[0] = batch_size;
		py::array arr(py::dtype(node.dtype_short_str), tmp_shape, input_buffer_list[i]);
		arr.inc_ref();
		oinput_list[i] = arr.ptr();
		/*NPY_TYPES np_type = NPY_INT;
		np_type = (NPY_TYPES)node.data_type;
		void* input_data_ptr = input_buffer_list[i];
		oinput_list[i] = PyArray_SimpleNewFromData(node.shape.size(), &node.shape[0], np_type, input_data_ptr);
		if (!oinput_list[i]) {
			has_error = true;
			log_err("transformer input %d data to arr failed", i);
			break;
		}
		PyArray_ENABLEFLAGS((PyArrayObject*)oinput_list[i], NPY_ARRAY_OWNDATA);*/
	}

	if (has_error) {
		code = -2;
	}
	else {
		code = engine->OnProcess(&pyresult, net_stage, input_num, oinput_list.data());
	}

	for (int i = 0; i < input_num; ++i) {
		SAFE_DECREF(oinput_list[i]);
	}
	if (code != 0) {
		return code;
	}
	
	int output_num = graph.output_.size();
	int infer_ouput_size = PyTuple_Size(pyresult);
	if (output_num != infer_ouput_size) {
		code = -3;
		log_err("output_num:%d does not match infer output num:%d\n", output_num, infer_ouput_size);
	}
	else {
		try {
			for (int k = 0; k < output_num; ++k) {
				char* out_result = (char*)final_result[k];

				auto oList = PyTuple_GetItem(pyresult, k);
				auto arr = py::cast <py::array>(oList);

				int element_size = arr.itemsize();
				for (int mm = 0, n_size = arr.size(); mm < n_size; ++mm) {
					memcpy(out_result + mm * element_size, arr.data(), element_size);
				}
			}
		}
		catch (std::exception& e) {
			code = -4;
			log_err("result cast failed , %s\n", e.what());
		}
	}
	SAFE_DECREF(pyresult);
	return code;
}


//for (int k = 0; k < output_num; ++k) {
	//	char* out_result = (char*)final_result[k];
	//	PyArrayObject* np_result = (PyArrayObject*)PyTuple_GetItem(pyresult, k);
	//	auto& node = graph.output_[k];

	//	int ndim = PyArray_NDIM(np_result);
	//	npy_intp* dims = PyArray_DIMS(np_result);

	//	auto strides = PyArray_STRIDES(np_result);
	//	char* data = (char*)PyArray_BYTES(np_result);
	//	auto np_type = PyArray_TYPE(np_result);


	//	int data_len = Get_dsize_by_type(np_type);
	//	if (data_len != Get_dsize_by_type(node.data_type)) {
	//		if (np_type != node.data_type) {
	//			log_err("infer graph output data type %d does not match the data type of config graph %d\n", np_type, node.data_type);
	//			code = -4;
	//			return code;
	//		}
	//	}
	//	int rows = (int)dims[0];
	//	if (ndim == 1) {
	//		for (int i = 0; i < rows; i++) {
	//			//auto & value = *(float *)(data + i);
	//			memcpy(out_result + i * data_len,
	//				data + i, data_len);
	//		}
	//	}
	//	else if (ndim == 2) {
	//		int columns = dims[1];
	//		for (int i = 0; i < rows; i++) {
	//			for (int j = 0; j < columns; j++) {
	//				/*auto & value = *(float *)(data + i * strides[0] + j * strides[1]);
	//				printf("(%d,%d) value=%f\n", i,j,value);*/
	//				memcpy(out_result + (i * columns + j) * data_len,
	//					data + i * strides[0] + j * strides[1],
	//					data_len);
	//			}
	//		}
	//	}
	//	else if (ndim == 3) {
	//		int columns = dims[1];
	//		int channel = dims[2];
	//		for (int i = 0; i < rows; i++) {
	//			for (int j = 0; j < columns; j++) {
	//				for (int k = 0; k < channel; ++k) {
	//					//auto & value = *(float *)(data + i * strides[0] + j * strides[1] + k * strides[2]);
	//					//printf("value=%f\n", value);
	//					memcpy(out_result + (i * columns * channel + j * channel + k) * data_len,
	//						data + i * strides[0] + j * strides[1] + k * strides[2],
	//						data_len);
	//				}
	//			}
	//		}
	//	}
	//	else {
	//		log_err("not support output shape %lld", node.shape.size());
	//		code = -6;
	//		return code;
	//	}
	//}
















//SDK_HANDLE_CC sdk_new_cc(const char* json_string) {
//	SDK_HANDLE_CC resouce_id = 0;
//	if (!json_string) return resouce_id;
//	log_info("%s\n", json_string);
//	rapidjson::Document d;
//	d.Parse(json_string);
//	if (d.HasParseError()) {
//		log_err("%s errCode: %d , offset: %d , parse json failed\n", __FUNCTION__,d.GetParseError(), d.GetErrorOffset());
//		return resouce_id;
//	}
//	log_info("%s parse json ok\n",__FUNCTION__);
//
//	S_aes_option aes_conf;
//	int use_saved_model = 0;
//	std::vector<std::string>vec_pb_tags;
//	std::string signature_key = "serving_default";
//
//	int device_id = 0, engine_version = 1, enable_graph = 0, model_type = 0;
//	int use_fastertransformer = 0;
//	try {
//		do {
//			std::string model_dir = d["model_dir"].GetString();
//			int log_level = 8;
//			if (d.HasMember("log_level"))
//				log_level = d["log_level"].GetInt();
//			the_config.log_level = log_level;
//			int engine_type = 0;
//			if (d.HasMember("engine"))
//				engine_type = d["engine"].GetInt();
//
//			if (engine_type < 0 && engine_type > 2) {
//				log_err("engine must be [0-2]\n");
//				break;
//			}
//
//			if (d.HasMember("aes")) {
//				auto aes = d["aes"].GetObject();
//				if (aes.HasMember("use")) {
//					aes_conf.use = aes["use"].GetInt();
//					if (aes_conf.use) {
//						if (!aes.HasMember("key") || !aes.HasMember("iv")) {
//							log_err("%s aes missing key ,iv\n", __FUNCTION__);
//							break;
//						}
//						auto key = aes["key"].GetArray();
//						auto iv = aes["iv"].GetArray();
//						if (key.Size() != 16 || iv.Size() != 16) {
//							log_err("%s aes missing key,iv\n", __FUNCTION__);
//							break;
//						}
//						for (int i = 0; i < 16;++i) {
//							aes_conf.key[i] = key[i].GetInt();
//							aes_conf.iv[i] = iv[i].GetInt();
//						}
//					}
//				}
//				else {
//					aes_conf.use = 0;
//				}
//			}
//
//
//			if (d.HasMember("device_id"))
//				device_id = d["device_id"].GetInt();
//
//			PyObject* oConfigProto = NULL;
//			if (engine_type == 0) {
//				oConfigProto = PyDict_New();
//
//				if (!oConfigProto) {
//					log_err("%s PyDict_New failed\n", __FUNCTION__);
//					break;
//				}
//				auto tf = d["tf"].GetObject();
//				model_type = tf["model_type"].GetInt();
//				engine_version = tf["engine_version"].GetInt();
//
//				if (tf.HasMember("fastertransformer")) {
//					auto  fastertransformer = tf["fastertransformer"].GetObject();
//					use_fastertransformer = fastertransformer["use"].GetInt();
//				}
//
//				if (tf.HasMember("saved_model")) {
//					auto saved_model = tf["saved_model"].GetObject();
//					use_saved_model = saved_model["use"].GetInt();
//					if (use_saved_model) {
//						if (saved_model.HasMember("signature_key")) {
//							signature_key = saved_model["signature_key"].GetString();
//						}
//						auto arr = saved_model["tags"].GetArray();
//						for (auto b = arr.Begin(); b != arr.End(); ++b) {
//							vec_pb_tags.push_back(b->GetString());
//						}
//					}
//				}
//				auto jd_ConfigProto = tf["ConfigProto"].GetObject();
//				int i_log_device_placement = 0;
//				int i_allow_soft_placement = 1;
//
//				if (jd_ConfigProto["log_device_placement"].IsBool()) {
//					i_log_device_placement = jd_ConfigProto["log_device_placement"].GetBool();
//				}
//				else if (jd_ConfigProto["log_device_placement"].IsInt()) {
//					i_log_device_placement = jd_ConfigProto["log_device_placement"].GetInt();
//				}
//
//				if (jd_ConfigProto["allow_soft_placement"].IsBool()) {
//					i_allow_soft_placement = jd_ConfigProto["allow_soft_placement"].GetBool();
//				}
//				else if (jd_ConfigProto["allow_soft_placement"].IsInt()) {
//					i_allow_soft_placement = jd_ConfigProto["allow_soft_placement"].GetInt();
//				}
//
//				my_PyDict_SetItemString(oConfigProto, "log_device_placement", PyBool_FromLong(i_log_device_placement));
//				my_PyDict_SetItemString(oConfigProto, "allow_soft_placement", PyBool_FromLong(i_allow_soft_placement));
//
//
//				auto ogpu_options = PyDict_New();
//				int i_allow_growth = 1;
//				if (jd_ConfigProto["gpu_options"]["allow_growth"].IsBool()) {
//					i_allow_growth = jd_ConfigProto["gpu_options"]["allow_growth"].GetBool();
//				}
//				else if (jd_ConfigProto["gpu_options"]["allow_growth"].IsInt()) {
//					i_allow_growth = jd_ConfigProto["gpu_options"]["allow_growth"].GetInt();
//				}
//
//				my_PyDict_SetItemString(ogpu_options, "allow_growth", PyBool_FromLong(i_allow_growth));
//				my_PyDict_SetItemString(oConfigProto, "gpu_options", ogpu_options);
//			}
//			else if (engine_type == 1) {
//				auto onnx = d["onnx"].GetObject();
//				
//				if (onnx.HasMember("engine_version"))
//					engine_version = onnx["engine_version"].GetInt();
//			}
//			else if (engine_type == 2) {
//				auto trt = d["trt"].GetObject();
//				if (trt.HasMember("engine_version"))
//					engine_version = trt["engine_version"].GetInt();
//				if (trt.HasMember("enable_graph"))
//					enable_graph = trt["enable_graph"].GetInt();
//			}
//			
//
//			auto jd_graph = d["graph"].GetArray();
//			int net_size = jd_graph.Size();
//			std::vector<S_my_net_graph>net_inf_graph;
//			net_inf_graph.resize(net_size);
//
//			bool berr = false;
//			int i = 0;
//			for (auto it = jd_graph.Begin(); it != jd_graph.End(); ++it, ++i) {
//				auto & curr_net_graph = net_inf_graph[i];
//				auto sub_net_graph = it->GetObject();
//
//				auto sub_input_graph = sub_net_graph["input"].GetArray();
//				auto sub_output_graph = sub_net_graph["output"].GetArray();
//
//
//				int sub_input_graph_size = sub_input_graph.Size();
//				int sub_output_graph_size = sub_output_graph.Size();
//
//
//				if (!sub_input_graph_size || !sub_output_graph_size) {
//					log_err("%s bad net_graph\n", __FUNCTION__);
//					berr = true;
//					break;
//				}
//
//				auto parse_graph_node = [](rapidjson::Value::Array & arr, std::vector<S_my_graph_node>& in_out) {
//					S_my_graph_node tmp_node;
//					for (auto it2 = arr.Begin(); it2 != arr.End(); ++it2) {
//						auto str_data_type = it2->GetObject()["data_type"].GetString();
//						auto data_shape = it2->GetObject()["shape"].GetArray();
//
//						tmp_node.data_type = Get_dtype_from_string(str_data_type);
//						tmp_node.name = it2->GetObject()["node"].GetString();
//
//						Get_dtype_string(tmp_node.data_type, tmp_node.dtype_short_str, tmp_node.dtype_long_str);
//
//						tmp_node.shape.clear();
//						for (auto it3 = data_shape.Begin(); it3 != data_shape.End(); ++it3) {
//							tmp_node.shape.push_back(it3->GetInt());
//						}
//						in_out.push_back(tmp_node);
//					}
//					return true;
//				};
//			
//				if (!parse_graph_node(sub_input_graph, curr_net_graph.input_)) {
//					log_err("%s parse_graph_node input failed\n", __FUNCTION__);
//					berr = true;
//					break;
//				}
//				curr_net_graph.oInput_.resize(sub_input_graph_size);
//				if (!parse_graph_node(sub_output_graph, curr_net_graph.output_)) {
//					log_err("%s parse_graph_node output failed\n", __FUNCTION__);
//					berr = true;
//					break;
//				}
//				curr_net_graph.oOutput_.resize(sub_output_graph_size);
//			}
//			if (berr) {
//				break;
//			}
//			log_info("model_dir %s,engine %d,engine_version %d,device_id=%d,ase.use=%d\n",
//				model_dir.c_str(), engine_type, engine_version, device_id,
//				aes_conf.use);
//			if (engine_type == 0) {
//				if (engine_version == 1) {
//					C_tf_v1_resource* inst = new C_tf_v1_resource(model_dir, device_id, net_inf_graph, engine_version, &aes_conf);
//					if (inst) {
//						if (!inst->OnLoad()) {
//							delete inst;
//						}
//						else {
//							if (0 == inst->OnCreate(model_type, oConfigProto, use_fastertransformer, use_saved_model, vec_pb_tags, signature_key)) {
//								resouce_id = (long long)inst;
//							}
//							else {
//								delete inst;
//							}
//						}
//					}
//				}
//				else {
//					C_tf_v2_resource* inst = new C_tf_v2_resource(model_dir, device_id, net_inf_graph, engine_version, &aes_conf);
//					if (inst) {
//						if (!inst->OnLoad()) {
//							delete inst;
//						}
//						else {
//							if (use_saved_model == 0) {
//								use_saved_model = 1;
//							}
//							if (0 == inst->OnCreate(model_type, use_saved_model, vec_pb_tags, signature_key)) {
//								resouce_id = (long long)inst;
//							}
//							else {
//								delete inst;
//							}
//
//						}
//					}
//				}
//			}
//			else if (engine_type == 1) {
//				C_onnx_resource* inst = new C_onnx_resource(model_dir, device_id, net_inf_graph, engine_version, &aes_conf);
//				if (inst) {
//					if (!inst->OnLoad()) {
//						delete inst;
//					}
//					else {
//						if (0 == inst->OnCreate()) {
//							resouce_id = (long long)inst;
//						}
//						else {
//							delete inst;
//						}
//
//					}
//				}
//			}
//			else if (engine_type == 2) {
//				C_trt_resource* inst = new C_trt_resource(model_dir, device_id, net_inf_graph, engine_version, &aes_conf);
//				if (inst) {
//					if (!inst->OnLoad()) {
//						delete inst;
//					}
//					else {
//						if (use_saved_model == 0) {
//							use_saved_model = 1;
//						}
//						if (0 == inst->OnCreate(enable_graph)) {
//							resouce_id = (long long)inst;
//						}
//						else {
//							delete inst;
//						}
//					}
//				}
//			}
//			
//		} while (0);
//	}
//	catch (std::exception & e) {
//		log_err("%s %s\n",__FUNCTION__,e.what());
//	}
//	return resouce_id;
//}