#include "onnx_csdk.h"

/*providers = [
	('CUDAExecutionProvider', {
		'device_id': 0,
		'arena_extend_strategy' : 'kNextPowerOfTwo',
		'gpu_mem_limit' : 2 * 1024 * 1024 * 1024,
		'cudnn_conv_algo_search' : 'EXHAUSTIVE',
		'do_copy_in_default_stream' : True,
		}),
		'CPUExecutionProvider',
]*/


py::object C_onnx_resource::ms_onnxruntime = py::none();
py::object C_onnx_resource::ms_onnxruntime_dict = py::none();;


int C_onnx_resource::onnx_Session(PyObject* oRead) {
	log_debug("%s...\n", __FUNCTION__);
	PyObject* oSessionClass = NULL, * oSessConstruct = NULL;
	int ret = -1;
	try {
		do {
			auto oOnnxruntimeDict = (*this)["onnxruntime"];
			PyObject* oSessionClass = my_PyDict_GetItemString(oOnnxruntimeDict, "InferenceSession");
			if (!oSessionClass) {
				PyErr_Print();
				log_err("%s get InferenceSession failed\n", __FUNCTION__);
				break;
			}
			oSessConstruct = PyInstanceMethod_New(oSessionClass);
			if (!oSessConstruct) {
				PyErr_Print();
				log_err("%s new InferenceSession failed\n", __FUNCTION__);
				break;
			}

			PyObject* pArgs = PyTuple_New(0);
			PyObject* kwargs = PyDict_New();
			PyDict_SetItem(kwargs, PyUnicode_FromString("path_or_bytes"), oRead);
			this->m_osession = PyObject_Call(oSessConstruct, pArgs, kwargs);
			Py_DECREF(pArgs);
			Py_DECREF(kwargs);
			if (!this->m_osession) {
				PyErr_Print();
				log_err("%s InferenceSession init failed\n", __FUNCTION__);
				break;
			}

			py::object osess = py::cast<py::object>(this->m_osession);
			if (py::hasattr(osess, "set_providers")) {
				log_info("set device_id %d...\n", m_device_id);
				auto set_providers = osess.attr("set_providers");
				if (m_ver[0] <= 1 && m_ver[1] <= 4) {
					if (m_device_id >= 0) {
						py::list param1(2);
						param1[0] = py::str("CUDAExecutionProvider");
						param1[1] = py::str("CPUExecutionProvider");
						set_providers(param1);
					}
					else if (m_device_id == -1) {
						py::list param1(1);
						param1[0] = py::str("CPUExecutionProvider");
						set_providers(param1);
					}
				}
				else {
					if (m_device_id >= 0) {
						py::list param1(2);
						param1[0] = py::str("CUDAExecutionProvider");
						param1[1] = py::str("CPUExecutionProvider");

						py::list param2(2);
						py::dict dict_param1, dict_param2;
						dict_param1["device_id"] = m_device_id;

						param2[0] = dict_param1;
						param2[1] = dict_param2;


						set_providers(param1, param2);
					}
					else if (m_device_id == -1) {
						py::list param1(1);
						param1[0] = py::str("CPUExecutionProvider");

						py::list param2(1);
						param2[0] = py::dict();

						set_providers(param1, param2);
					}
				}
			}
			
			log_info("load node from graph\n");
			if (py::hasattr(osess, "get_inputs")) {
				auto get_inputs = osess.attr("get_inputs");
				py::list inputs = get_inputs();
				for (int i = 0, N_size = inputs.size(); i < N_size; ++i) {
					log_info("input %d , name %s , type %s , shape %s\n", i, 
						py::str(inputs[i].attr("name")).operator std::string().c_str(),
						py::str(inputs[i].attr("type")).operator std::string().c_str(),
						py::str(inputs[i].attr("shape")).operator std::string().c_str()
					);
				}
			}

			if (py::hasattr(osess, "get_outputs")) {
				auto get_outputs = osess.attr("get_outputs");
				py::list outputs = get_outputs();
				for (int i = 0, N_size = outputs.size(); i < N_size; ++i) {
					log_info("output %d , name %s , type %s , shape %s\n",i, 
						py::str(outputs[i].attr("name")).operator std::string().c_str(),
						py::str(outputs[i].attr("type")).operator std::string().c_str(),
						py::str(outputs[i].attr("shape")).operator std::string().c_str()
					);
				}
			}
			ret = 0;
		} while (0);
	}
	catch (std::exception& e) {
		ret = -1;
		log_err("%s %s\n",__FUNCTION__,e.what());
	}
	

	SAFE_DECREF(oSessConstruct);
	SAFE_DECREF(oSessionClass);
	return ret;
}


int C_onnx_resource::onnx_Session_ex(PyObject* oRead,int enable_tensorrt) {
	log_debug("%s...\n", __FUNCTION__);
	PyObject* oSessionClass = NULL, * oSessConstruct = NULL;
	int ret = -1;
	try {
		do {
			auto oOnnxruntimeDict = (*this)["onnxruntime"];
			PyObject* oSessionClass = my_PyDict_GetItemString(oOnnxruntimeDict, "InferenceSession");
			if (!oSessionClass) {
				PyErr_Print();
				log_err("%s get InferenceSession failed\n", __FUNCTION__);
				break;
			}
			oSessConstruct = PyInstanceMethod_New(oSessionClass);
			if (!oSessConstruct) {
				PyErr_Print();
				log_err("%s new InferenceSession failed\n", __FUNCTION__);
				break;
			}


			//�����Կ�id

			py::tuple providers_param(2);
			if (m_device_id >= 0) {
				py::list param1(0);
				
				if (enable_tensorrt) {
					param1.append(py::str("TensorrtExecutionProvider"));
				}
				
				param1.append(py::str("CUDAExecutionProvider"));
				param1.append(py::str("CPUExecutionProvider"));

				py::list param2(0);
				py::dict dict_param1, dict_param2, dict_param3;
				dict_param1["device_id"] = m_device_id;
				dict_param2["device_id"] = m_device_id;

				if (enable_tensorrt) {
					param2.append(dict_param1);
				}
				param2.append(dict_param2);
				param2.append(dict_param3);

				param1.inc_ref();
				param2.inc_ref();

				providers_param[0] = param1;
				providers_param[1] = param2;
			}
			else {
				py::list param1(1);
				param1[0] = py::str("CPUExecutionProvider");

				py::list param2(1);
				param2[0] = py::dict();

				param1.inc_ref();
				param2.inc_ref();

				providers_param[0] = param1;
				providers_param[1] = param2;
			}

			
			PyObject* pArgs = PyTuple_New(0);
			PyObject* kwargs = PyDict_New();
			PyDict_SetItem(kwargs, PyUnicode_FromString("path_or_bytes"), oRead);
			PyDict_SetItem(kwargs, PyUnicode_FromString("providers"), providers_param[0].ptr());
			PyDict_SetItem(kwargs, PyUnicode_FromString("provider_options"), providers_param[1].ptr());
			this->m_osession = PyObject_Call(oSessConstruct, pArgs, kwargs);
			Py_DECREF(pArgs);
			Py_DECREF(kwargs);
			if (!this->m_osession) {
				PyErr_Print();
				log_err("%s InferenceSession init failed\n", __FUNCTION__);
				break;
			}

			py::object osess = py::cast<py::object>(this->m_osession);


			log_info("load node from graph\n");
			if (py::hasattr(osess, "get_inputs")) {
				auto get_inputs = osess.attr("get_inputs");
				py::list inputs = get_inputs();
				for (int i = 0, N_size = inputs.size(); i < N_size; ++i) {
					log_info("input %d , name %s , type %s , shape %s\n", i,
						py::str(inputs[i].attr("name")).operator std::string().c_str(),
						py::str(inputs[i].attr("type")).operator std::string().c_str(),
						py::str(inputs[i].attr("shape")).operator std::string().c_str()
					);
				}
			}

			if (py::hasattr(osess, "get_outputs")) {
				auto get_outputs = osess.attr("get_outputs");
				py::list outputs = get_outputs();
				for (int i = 0, N_size = outputs.size(); i < N_size; ++i) {
					log_info("output %d , name %s , type %s , shape %s\n", i,
						py::str(outputs[i].attr("name")).operator std::string().c_str(),
						py::str(outputs[i].attr("type")).operator std::string().c_str(),
						py::str(outputs[i].attr("shape")).operator std::string().c_str()
					);
				}
			}
			ret = 0;
		} while (0);
	}
	catch (std::exception& e) {
		ret = -1;
		log_err("%s %s\n", __FUNCTION__, e.what());
	}


	SAFE_DECREF(oSessConstruct);
	SAFE_DECREF(oSessionClass);
	return ret;
}







int C_onnx_resource::onnx_make_feeds_fetch() {
	log_debug("%s...\n", __FUNCTION__);
	if (this->m_net_graph.size() < 0) {
		return -1;
	}

	for (auto & it : this->m_net_graph) {
		for (int i = 0; i< int(it.input_.size()); ++i) {
			auto & oo = it.oInput_[i];
			oo = PyUnicode_FromString(it.input_[i].name.c_str());
		}

		for (int i = 0; i< int(it.output_.size()); ++i) {
			auto & oo = it.oOutput_[i];
			oo = PyUnicode_FromString(it.output_[i].name.c_str());
		}
	}
	return 0;
}


int C_onnx_resource::OnCreate(int enable_tensorrt){
	log_debug("%s...\n", __FUNCTION__);
	PyObject* oRead = NULL;
	std::string sread;
	if (read_file(this->m_model_dir.c_str(), sread) <= 0) {
		return -1;
	}
	if (m_aes_conf.use) {
		std::string sread_plain;
		if (0 != tk_aes_decode((uint8_t*)sread.c_str(), sread.size(), sread_plain, m_aes_conf.key, m_aes_conf.iv)) {
			log_err("aes decode failed");
			return -1;
		}
		oRead = Py_BuildValue("y#", sread_plain.c_str(), sread_plain.size());
	}
	else {
		oRead = Py_BuildValue("y#", sread.c_str(), sread.size());
	}
	if (!oRead) {
		return -1;
	}
	
	if (m_ver[0] <= 1 && m_ver[1] < 9) {
		if (0 != onnx_Session(oRead)) {
			log_err("onnx_Session failed\n");
			return -1;
		}
	}
	else {
		if (0 != onnx_Session_ex(oRead, enable_tensorrt)) {
			log_err("onnx_Session failed\n");
			return -1;
		}
	
	}
	


	if (0 != onnx_make_feeds_fetch()) {
		log_err("onnx_make_feeds_fetch failed\n");
		PyErr_Print();
		return -1;
	}

	if (!this->load_sub_func()) {
		log_err("load_sub_func failed\n");
		PyErr_Print();
		return -1;
	}
	return 0;
}


int C_onnx_resource::OnProcess(PyObject**result, int stage, int input_num_, PyObject** inputs_) {

	if (stage > this->m_net_graph.size()) {
		log_err("%s bad input stage %d\n", __FUNCTION__, stage);
		return -1;
	}
	auto net_inf_stage = this->m_net_graph[stage];
	if (net_inf_stage.oInput_.size() != input_num_) {
		log_err("%s bad input data num\n", __FUNCTION__);
		return -1;
	}

	auto & ofetch = this->m_ofetchs[stage];
	for (int i = 0; i < input_num_; ++i) {
		PyDict_SetItem(m_ofeed_dict, net_inf_stage.oInput_[i], inputs_[i]);
	}

	PyObject *pArgs = PyTuple_New(2);
	PyTuple_SetItem(pArgs, 0, ofetch);
	PyTuple_SetItem(pArgs, 1, m_ofeed_dict);

	auto r = PyObject_CallObject(this->m_orun, pArgs);
	PyDict_Clear(m_ofeed_dict);

	Py_INCREF(ofetch);
	Py_INCREF(m_ofeed_dict);
	Py_DECREF(pArgs);
	if (r == NULL) {
		PyErr_Print();
		return -1;
	}
	*result = r;
	return 0;
}