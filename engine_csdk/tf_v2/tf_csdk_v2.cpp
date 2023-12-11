#include "tf_csdk_v2.h"


PyObject* GET_TF_V2_ATTR(C_tf_v2_resource* resource,const char* name) {
	auto oTfModule = (*resource)["tensorflow"];
	if (resource->m_is_kernel_object) {
		return my_PyObject_GetAttrString(oTfModule, name);
	}
	return my_PyDict_GetItemString(oTfModule, name);
}


int C_tf_v2_resource::load_graph_by_saved_model_pb_v2(std::string& signature_key, std::vector<std::string>& vec_pb_tags) {
	log_debug("%s...\n", __FUNCTION__);
	auto oTfModule = (*this)["tensorflow"];
	bool berr = false;
	PyObject* osaved_model = NULL, *oload = NULL, *osvmod = NULL, *sig = NULL;
	do {
		osaved_model = my_PyDict_GetItemString(oTfModule, "saved_model");
		if (!osaved_model) {
			log_err("%s tf.saved_model failed\n",__FUNCTION__);
			berr = true;
			break;
		}
		oload = my_PyObject_GetAttrString(osaved_model, "load");
		if (!oload) {
			log_err("%s tf.saved_model.load failed !\n", __FUNCTION__);
			berr = true;
			break;
		}
		{
			auto oArgs = PyTuple_New(0);
			auto okwargs = PyDict_New();
			PyDict_SetItem(okwargs, PyUnicode_FromString("export_dir"), PyUnicode_FromString(m_model_dir.c_str()));
			if (vec_pb_tags.size() > 0) {
				auto param_list = PyList_New(vec_pb_tags.size());
				for (size_t i = 0, N_size = vec_pb_tags.size(); i < N_size; ++i) {
					PyList_SetItem(param_list, i, PyUnicode_FromString(vec_pb_tags[i].c_str()));
				}
				PyDict_SetItem(okwargs, Py_BuildValue("s", "tags"), param_list);
			}
			osvmod = PyObject_Call(oload, oArgs, okwargs);
			Py_DECREF(oArgs);
			if (!osvmod) {
				log_err("model dir %s\n", m_model_dir.c_str());
				log_err("%s saved_model.load failed !!\n", __FUNCTION__);
				berr = true;
				break;
			}
		}
		
		sig = my_PyObject_GetAttrString(osvmod, "signatures");
		if (!sig) {
			log_err("%s signatures failed\n", __FUNCTION__);
			berr = true;
			break;
		}

		auto getitem = my_PyObject_GetAttrString(sig, "__getitem__");
		if (!getitem) {
			log_err("%s __getitem__ failed\n", __FUNCTION__);
			berr = true;
			break;
		}
		auto oArgs = PyTuple_New(1);
		PyTuple_SetItem(oArgs, 0, PyUnicode_FromString(signature_key.c_str()));
		this->m_infer_func = PyObject_CallObject(getitem, oArgs);
		Py_DECREF(oArgs);
		Py_DECREF(getitem);
		
		if (!this->m_infer_func) {
			log_err("%s infer_func load failed\n",__FUNCTION__);
			berr = true;
			break;
		}

		//PyObject_Print(this->m_infer_func, stdout, 0);
	} while (0);

	m_map.insert(std::make_pair("saved_model", osaved_model));
	m_map.insert(std::make_pair("saved_model.onload", oload));
	m_map.insert(std::make_pair("svmod", osvmod));
	m_map.insert(std::make_pair("signatures", sig));

	/*SAFE_DECREF(osaved_model);
	SAFE_DECREF(oload);
	SAFE_DECREF(osvmod);
	SAFE_DECREF(sig);*/
	if (berr) {
		PyErr_Print();
		return -1;
	}
	return 0;
}





int C_tf_v2_resource::OnProcess(PyObject** result, int stage, int input_num_, PyObject** inputs_) {
	auto net_inf_stage = this->m_net_graph[stage];
	if (net_inf_stage.oInput_.size() != input_num_) {
		log_err("%s bad input num, graph num: %lld , input_num: %lld\n", __FUNCTION__, net_inf_stage.oInput_.size(), input_num_);
		return -1;
	}
	auto oConstant = (*this)["constant"];
	m_vec_inputs.resize(input_num_);

	py::dict dict_param;
	auto cur_input = net_inf_stage.input_;
	for (int i = 0; i < input_num_; ++i) {
		auto& input_ids = inputs_[i];
		//int size = (int)PyList_GET_SIZE(input_ids);
		/*PyTuple_SetItem(shape, 0, Py_BuildValue("i", size));
		PyTuple_SetItem(shape, 1, Py_BuildValue("i", this->m_max_seq_length));*/
		//PyDict_SetItemString(oDict, "shape", shape);

		auto oData_type = (*this)[cur_input[i].dtype_long_str.c_str()];
		if (oData_type) {
			Py_INCREF(oData_type);
			dict_param["dtype"] = oData_type;
		}

		dict_param["value"] = input_ids;
		m_vec_inputs[i] = PyObject_Call(oConstant, m_orun_args_0, dict_param.ptr());
		dict_param.clear();
		if (!m_vec_inputs[i]) {
			PyErr_Print();
			return -1;
		}
		PyDict_SetItemString(m_ofeed_dict, net_inf_stage.input_[i].name.c_str(), m_vec_inputs[i]);
	}
	//PyDict_SetItemString(this->m_ofeed, "input_ids", my_input_ids);
	//PyDict_SetItemString(this->m_ofeed, "input_mask", my_input_mask);


	auto r = PyObject_Call(this->m_infer_func, m_orun_args_0, m_ofeed_dict);

	PyDict_Clear(m_ofeed_dict);
	if (r == NULL) {
		PyErr_Print();
		return -1;
	}

	if (PyDict_Check(r)) {
		auto olist = PyList_New(net_inf_stage.output_.size());
		Py_ssize_t i = 0;
		for (auto& item : net_inf_stage.output_) {
			PyObject* o = PyDict_GetItemString(r, item.name.c_str());
			if (o) {
				Py_INCREF(o);
			}
			else {
				Py_INCREF(Py_None);
				o = Py_None;
			}
			PyList_SetItem(olist, i++, o);
		}
		*result = olist;
		Py_DECREF(r);
	}
	else {
		*result = r;
	}
	return 0;
}

int C_tf_v2_resource::OnCreate(int model_type,  int use_saved_model, std::vector<std::string>& vec_pb_tags, std::string& signature_key){

	if (0 != load_graph_by_saved_model_pb_v2(signature_key, vec_pb_tags)){
		log_debug("%s load_graph_by_saved_model_pb_v2 failed\n", __FUNCTION__);
		PyErr_Print();
		return -1;
	}
	if (!load_sub_func()) {
		log_err("load_sub_func failed");
		PyErr_Print();
		return -1;
	}
	return 0;
}