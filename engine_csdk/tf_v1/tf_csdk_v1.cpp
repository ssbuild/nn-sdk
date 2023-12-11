#include "tf_csdk_v1.h"
//int tk_ConfigProto(C_tf_v1_resource* resource, PyObject**result) {
//	auto oTfModule = (*resource)["tensorflow"];
//	//gpu_options = tf.GPUOptions(allow_growth = True)
//	PyObject* oGpu_options = NULL;
//	{
//		PyObject *pArgs = PyTuple_New(0);
//		PyObject* pArgsD = PyDict_New();
//		PyDict_SetItemString(pArgsD, "allow_growth", Py_BuildValue("i", 1));
//
//		auto gpu_options_cls = PyObject_GetAttrString(oTfModule, "GPUOptions");
//		PyObject *pthis_gpu_options = PyInstanceMethod_New(gpu_options_cls);
//		Py_DECREF(gpu_options_cls);
//
//		oGpu_options = PyObject_Call(pthis_gpu_options, pArgs, pArgsD);
//		Py_DECREF(pArgs);
//		Py_DECREF(pArgsD);
//		Py_DECREF(pthis_gpu_options);
//	}
//
//	if (!oGpu_options) {
//		PyErr_Print();
//		return -1;
//	}
//
//
//	PyObject* oConfigProto = PyObject_GetAttrString(oTfModule, "ConfigProto");
//	PyObject *pthis_ConfigProto = PyInstanceMethod_New(oConfigProto);
//	Py_DECREF(oConfigProto);
//
//
//	PyObject *pArgs = PyTuple_New(0);
//	PyObject* pArgsD = PyDict_New();
//
//
//	PyDict_SetItemString(pArgsD, "log_device_placement", Py_BuildValue("i", 0));
//	PyDict_SetItemString(pArgsD, "allow_soft_placement", Py_BuildValue("i", 1));
//	PyDict_SetItemString(pArgsD, "gpu_options", oGpu_options);
//	
//	*result = PyObject_Call(pthis_ConfigProto, pArgs, pArgsD);
//	Py_DECREF(pArgs);
//	Py_DECREF(pArgsD);
//	Py_DECREF(pthis_ConfigProto);
//	if (*result == NULL) {
//		PyErr_Print();
//		return -1;
//	}
//	return 0;
//}


PyObject* GET_TF_V1_ATTR(C_tf_v1_resource* resource,const char* name) {
	auto oTfModule = (*resource)["tensorflow"];
	if (resource->m_is_kernel_object) {
		return my_PyObject_GetAttrString(oTfModule, name);
	}
	return my_PyDict_GetItemString(oTfModule, name);
}


int C_tf_v1_resource::tk_ConfigProto_ex(PyObject* oConfigProto_data, PyObject**result) {
	log_debug("%s...\n", __FUNCTION__);
	try {
		PyObject* oConfigProto = NULL;
		oConfigProto = GET_TF_V1_ATTR(this,"ConfigProto");
		if (!oConfigProto) {
			log_err("get function ConfigProto failed\n");
			PyErr_Print();
			return -1;
		}
		PyObject* pthis_ConfigProto = PyInstanceMethod_New(oConfigProto);
		Py_DECREF(oConfigProto);

		PyObject *pArgs = PyTuple_New(0);
		*result = PyObject_Call(pthis_ConfigProto, pArgs, oConfigProto_data);
		Py_DECREF(pArgs);
		Py_DECREF(pthis_ConfigProto);
		if (*result == NULL) {
			log_err("tf.ConfigProto  failed\n");
			PyErr_Print();
			return -1;
		}
		return 0;
	}
	catch (std::exception & e) {
		log_err("err: %s\n",e.what());
	}
	return -1;
}

int C_tf_v1_resource::tf_Session(PyObject* oConfig) {
	log_debug("%s...\n", __FUNCTION__);

	PyObject* oSession = GET_TF_V1_ATTR(this, "Session");
	if (!oSession) {
		log_err("get function Session failed\n");
		PyErr_Print();
		return -1;
	}
	auto inst = PyInstanceMethod_New(oSession);
	Py_DECREF(oSession);

	PyObject *pArgs = PyTuple_New(0);
	PyObject *pDict = PyDict_New();

	if (oConfig) {
		PyDict_SetItemString(pDict, "config", oConfig);
	}
	
	this->m_osession = PyObject_Call(inst, pArgs, pDict);

	Py_DECREF(pArgs);
	Py_DECREF(pDict);
	Py_DECREF(inst);
	
	if (!this->m_osession) {
		log_err("tf.Session failed\n");
		PyErr_Print();
		return -1;
	}
	return 0;
}


int C_tf_v1_resource::tf_load_graph_by_ckpt() {
	log_debug("%s...\n", __FUNCTION__);
	int ret = -1;
	auto oTfModule = (*this)["tensorflow"];

	auto otrain = GET_TF_V1_ATTR(this, "train");
	if (!otrain) {
		log_err("%s get item train failed\n",__FUNCTION__);
		return -1;
	}
	PyObject* oimport_meta_graph = NULL;
	if (m_is_kernel_object) {
		oimport_meta_graph = PyObject_GetAttrString(otrain, "import_meta_graph");
		Py_DECREF(otrain);
	}
	else {
		PyObject* tmp_func_dict = PyModule_GetDict(otrain);
		Py_DECREF(otrain);
		if (!tmp_func_dict) {
			log_err("%s object train PyModule_GetDict failed\n", __FUNCTION__);
			return -1;
		}
		oimport_meta_graph = my_PyDict_GetItemString(tmp_func_dict, "import_meta_graph");
		Py_DECREF(tmp_func_dict);
	}
	
	
	if (!oimport_meta_graph) {
		PyErr_Print();
		log_err("get tf.import_meta_graph failed\n");
		return false;
	}

	char meta_filename[256];
	strcpy(meta_filename, m_model_dir.c_str());
	strcat(meta_filename,".meta");
	auto oMeta_file = Py_BuildValue("s", meta_filename);



	PyObject* oFuncRestore = NULL,*oTemp_Default_graph = NULL,*restore_ret = NULL;
	PyObject* pArgs = PyTuple_New(1);
	PyTuple_SetItem(pArgs, 0, oMeta_file);

	auto saver = PyObject_CallObject(oimport_meta_graph, pArgs);
	Py_DECREF(pArgs);
	if (!saver) {
		goto End;
	}
	oFuncRestore = my_PyObject_GetAttrString(saver, "restore");
	if (!oFuncRestore) {
		goto End;
	}
	Py_INCREF(m_osession);

	pArgs = PyTuple_New(2);
	oMeta_file = Py_BuildValue("s", m_model_dir.c_str());
	PyTuple_SetItem(pArgs, 0, this->m_osession);
	PyTuple_SetItem(pArgs, 1, oMeta_file);

	restore_ret = PyObject_CallObject(oFuncRestore, pArgs);
	Py_DECREF(pArgs);
	if (!restore_ret) {
		goto End;
	}
	SAFE_DECREF(restore_ret);
	{
		oTemp_Default_graph = GET_TF_V1_ATTR(this, "get_default_graph");
		if (!oTemp_Default_graph) {
			log_err("%s tf.get_default_graph failed\n",__FUNCTION__);
			goto End;
		}
		this->m_ograph = PyObject_CallObject(oTemp_Default_graph, NULL);
		Py_DECREF(oTemp_Default_graph);
	}

	if (this->m_ograph == NULL) {
		PyErr_Print();
		goto End;
	}
	ret = 0;
End:
	SAFE_DECREF(oFuncRestore);
	SAFE_DECREF(saver);
	Py_DECREF(oimport_meta_graph);
	
	
	return ret;
}

int C_tf_v1_resource::load_graph_by_pb(PyObject*oRead){
	log_debug("%s...\n", __FUNCTION__);
	PyObject *pthis_GraphDef = NULL;
	{
		PyObject* oGraphDef = GET_TF_V1_ATTR(this, "GraphDef");
		auto inst = PyInstanceMethod_New(oGraphDef);
		pthis_GraphDef = PyObject_CallObject(inst, NULL);
		Py_DECREF(inst);
		Py_DECREF(oGraphDef);
		if (!pthis_GraphDef) {
			log_err("%s tf.GraphDef failed\n",__FUNCTION__);
			PyErr_Print();
			return -1;
		}
	}

	{
		PyObject* oParseFromString = my_PyObject_GetAttrString(pthis_GraphDef, "ParseFromString");
		PyObject *pArgs = PyTuple_New(1);
		PyTuple_SetItem(pArgs, 0, oRead);
		PyObject* oTemp = PyObject_CallObject(oParseFromString, pArgs);
		Py_DECREF(pArgs);
		Py_DECREF(oParseFromString);

		if (oTemp) {
			Py_DECREF(oTemp);
		}
		else {
			log_err("%s tf.GraphDef.ParseFromString failed\n", __FUNCTION__);
			PyErr_Print();
			Py_DECREF(pthis_GraphDef);
			return -1;
		}
	}


	{
		
		PyObject* oImport_graph_def = GET_TF_V1_ATTR(this, "import_graph_def");

		PyObject* pArgs = PyTuple_New(1);
		PyTuple_SetItem(pArgs, 0, pthis_GraphDef);

		PyObject* pDict = PyDict_New();
		PyDict_SetItemString(pDict, "name", Py_BuildValue("s", ""));

		auto oTemp = PyObject_Call(oImport_graph_def, pArgs, pDict);
		Py_DECREF(oImport_graph_def);
		Py_DECREF(pDict);
		Py_DECREF(pArgs);
		if (oTemp) {
			Py_DECREF(oTemp);
		}
		else {
			log_err("%s tf.import_graph_def failed\n", __FUNCTION__);
			PyErr_Print();
			return -1;
		}
	}

	{
		PyObject* oDefault_graph = GET_TF_V1_ATTR(this, "get_default_graph");
		this->m_ograph = PyObject_CallObject(oDefault_graph, NULL);
		Py_DECREF(oDefault_graph);
	}

	if (this->m_ograph == NULL) {
		log_err("%s tf.get_default_graph failed\n", __FUNCTION__);
		PyErr_Print();
		return -1;
	}
	return 0;
}


//import tensorflow as tf
//sess = tf.Session()
//signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
//input_key = 'x_input'
//output_key = 'y_output'
//
//export_path = './savedmodel'
//meta_graph_def = tf.saved_model.loader.load(
//	sess,
//	[tf.saved_model.tag_constants.SERVING],
//	export_path)
//	signature = meta_graph_def.signature_def
//
//	x_tensor_name = signature[signature_key].inputs[input_key].name
//	y_tensor_name = signature[signature_key].outputs[output_key].name
//
//	x = sess.graph.get_tensor_by_name(x_tensor_name)
//	y = sess.graph.get_tensor_by_name(y_tensor_name)
//
//	y_out = sess.run(y, { x: 3.0 })
//
//	print(y_out)

int C_tf_v1_resource::load_graph_by_saved_model_pb(std::string & signature_key,std::vector<std::string>& vec_pb_tags) {
	log_debug("%s...\n", __FUNCTION__);
	auto oTfModule = (*this)["tensorflow"];
	bool berr = false;
	PyObject* osaved_model = NULL, *oload = NULL, *osvmod = NULL, *sig = NULL;
	do {
		osaved_model = my_PyDict_GetItemString(oTfModule, "saved_model");
		if (!osaved_model) {
			osaved_model = my_PyObject_GetAttrString(oTfModule, "saved_model");
			if (!osaved_model) {
				log_err("%s load saved_model failed\n", __FUNCTION__);
				berr = true;
				break;
			}
		}
		oload = my_PyObject_GetAttrString(osaved_model, "load");
		if (!oload) {
			log_err("%s saved_model.load failed !\n", __FUNCTION__);
			berr = true;
			break;
		}

		{
			auto oArgs = PyTuple_New(0);
			auto okwargs = PyDict_New();
			auto param_list = PyList_New(vec_pb_tags.size());
			for (size_t i = 0, N_size = vec_pb_tags.size(); i < N_size; ++i) {
				PyList_SetItem(param_list, i, PyUnicode_FromString(vec_pb_tags[i].c_str()));
			}

			PyDict_SetItem(okwargs, PyUnicode_FromString("export_dir"), PyUnicode_FromString(m_model_dir.c_str()));
			PyDict_SetItem(okwargs, PyUnicode_FromString("tags"), param_list);

			Py_IncRef(this->m_osession);
			PyDict_SetItem(okwargs, PyUnicode_FromString("sess"), this->m_osession);
			

			osvmod = PyObject_Call(oload, oArgs, okwargs);
			Py_DECREF(oArgs);
			Py_DECREF(okwargs);
			if (!osvmod) {
				log_err("%s call saved_model.load failed\n", __FUNCTION__);
				berr = true;
				break;
			}
		}

		this->m_ograph = my_PyObject_GetAttrString(m_osession, "graph");
		if (this->m_ograph == NULL) {
			log_err("%s get session graph failed\n", __FUNCTION__);
			PyErr_Print();
			return -1;
		}
		
		//tf1
		sig = my_PyObject_GetAttrString(osvmod, "signature_def");
		// tf2
		//sig = my_PyObject_GetAttrString(osvmod, "signatures");
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
		this->m_infer_signature = PyObject_CallObject(getitem, oArgs);
		Py_DECREF(oArgs);
		Py_DECREF(getitem);
		
		if (!this->m_infer_signature) {
			log_err("%s signature load failed\n",__FUNCTION__);
			berr = true;
			break;
		}
	} while (0);
	SAFE_DECREF(osaved_model);
	SAFE_DECREF(oload);
	SAFE_DECREF(osvmod);
	SAFE_DECREF(sig);
	if (berr) {
		PyErr_Print();
		return -1;
	}
	log_info("%s ok\n",__FUNCTION__);
	return 0;
}

int C_tf_v1_resource::tf_get_tensor_saved_model() {
	log_debug("%s...\n", __FUNCTION__);
	auto signature = py::cast<py::object>(this->m_infer_signature);
	py::dict sig_inputs = signature.attr("inputs");
	py::dict sig_outputs = signature.attr("outputs");

	if (sig_inputs.size() == 0 || sig_outputs.size() == 0) {
		log_err("signature has no inputs or outputs\n");
		return -1;
	}
	for (auto it = sig_inputs.begin(); it != sig_inputs.end();it++) {
		PyObject_Print(it->first.ptr(), stdout, 0);
		PyObject_Print(it->second.ptr(), stdout, 0);
	}

	for (auto it = sig_outputs.begin(); it != sig_outputs.end(); it++) {
		PyObject_Print(it->first.ptr(), stdout, 0);
		PyObject_Print(it->second.ptr(), stdout, 0);
	}
	//重写节点名字
	for (auto& it : this->m_net_graph) {

		//找到输入节点tenor
		for (int i = 0; i< int(it.input_.size()); ++i) {
			it.input_[i].name = py::str(sig_inputs[it.input_[i].name.c_str()].attr("name")).operator std::string();
		}
		//找到输出节点tenor
		for (int i = 0; i< int(it.output_.size()); ++i) {
			it.output_[i].name = py::str(sig_outputs[it.output_[i].name.c_str()].attr("name")).operator std::string();
		}
	}
	return 0;
}

/* tensorflow 1 load 输入输出节点 */
int C_tf_v1_resource::tf_get_tensor(){
	log_debug("%s...\n", __FUNCTION__);
	if (this->m_net_graph.size() < 0) {
		return -1;
	}
	PyObject* get_tensor_by_name = my_PyObject_GetAttrString(this->m_ograph, "get_tensor_by_name");
	if (get_tensor_by_name == NULL) {
		log_err("%s ograph get_tensor_by_name failed", __FUNCTION__);
		PyErr_Print();
		return -1;
	}

	for (auto & it : this->m_net_graph) {

		//找到输入节点tenor
		for (int i = 0; i< int(it.input_.size()); ++i) {
			auto & oo = it.oInput_[i];
			PyObject *pArgs = PyTuple_New(1);
			//PyTuple_SetItem(pArgs, 0, Py_BuildValue("s", "input_ids:0"));
			PyTuple_SetItem(pArgs, 0, Py_BuildValue("s", it.input_[i].name.c_str()));
			oo = PyObject_CallObject(get_tensor_by_name, pArgs);
			Py_DECREF(pArgs);
			if (oo == NULL) {
				log_err("%s get_tensor_by_name %s failed", __FUNCTION__, it.input_[i].name.c_str());
				PyErr_Print();
				Py_DECREF(get_tensor_by_name);
				return -1;
			}
		}
		//找到输出节点tenor
		for (int i = 0; i< int(it.output_.size()); ++i) {
			auto & oo = it.oOutput_[i];
			PyObject *pArgs = PyTuple_New(1);
			PyTuple_SetItem(pArgs, 0, Py_BuildValue("s", it.output_[i].name.c_str()));
			oo = PyObject_CallObject(get_tensor_by_name, pArgs);
			Py_DECREF(pArgs);
			if (oo == NULL) {
				log_err("%s get_tensor_by_name %s failed", __FUNCTION__, it.output_[i].name.c_str());
				PyErr_Print();
				Py_DECREF(get_tensor_by_name);
				return -1;
			}
		}
	}
	Py_DECREF(get_tensor_by_name);
	return 0;
}

int C_tf_v1_resource::OnProcess(PyObject**result,int stage,int input_num_, PyObject** inputs_) {
	if (stage > this->m_net_graph.size()) {
		log_err("%s bad input stage %d\n", __FUNCTION__, stage);
		return -1;
	}
	auto net_inf_stage = this->m_net_graph[stage];
	if (net_inf_stage.oInput_.size() != input_num_) {
		log_err("%s bad input data num\n",__FUNCTION__);
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


int C_tf_v1_resource::tf_reset_graph() {

	bool berr = false;
	do {
		auto oreset_default_graph = GET_TF_V1_ATTR(this, "reset_default_graph");
		if (!oreset_default_graph) {
			log_err("%s load reset_default_graph failed !\n", __FUNCTION__);
			berr = true;
			break;
		}

		auto oArgs = PyTuple_New(0);
		auto oempty = PyObject_CallObject(oreset_default_graph, oArgs);

		Py_DECREF(oArgs);
		if (oempty) {
			Py_DECREF(oempty);
		}
		Py_DECREF(oreset_default_graph);
	} while (false);
	return berr ? -1 : 0;
}


int C_tf_v1_resource::OnCreate(int model_type,
	int is_reset_graph,
	PyObject* oConfigProto,
	int  use_fastertransformer, 
	int use_saved_model,
	std::vector<std::string>& vec_pb_tags, std::string& signature_key) {


	log_debug("%s...\n", __FUNCTION__);
	if (use_fastertransformer) {
		try {
			log_info("start fastertransformer...\n");
			auto py_fastertransformer = py::module_::import("fastertransformer");
			auto ft_path = py::str(py_fastertransformer.attr("__file__")).operator std::string();
#ifdef _WIN32
			ft_path = ft_path.substr(0, ft_path.rfind('\\'));
#else
			ft_path = ft_path.substr(0, ft_path.rfind('/'));
#endif // !_WIN32

			char path_file[256] = { 0 };
			snprintf(path_file, sizeof(path_file), "%s/libtf_fastertransformer.so",ft_path.c_str());
			if (access(path_file, 0) != 0) {
				snprintf(path_file, sizeof(path_file), "%s/libtf_bert.so", ft_path.c_str());
				if (access(path_file, 0) != 0) {
					log_err("load fastertransformer op failed\n");
					return -1;
				}
			}


			log_debug("load %s\n", path_file);
			auto tf = py::module_::import("tensorflow");
			auto load_op_library = tf.attr("load_op_library");
			load_op_library(py::str(path_file));
			log_info("start fastertransformer ok\n");
		}
		catch (std::exception& e) {
			log_err("import fastertransformer failed , %s\n", e.what());
			return -1;
		}
	}


	PyObject* oConfig = NULL;
	if (0 != tk_ConfigProto_ex(oConfigProto, &oConfig)) {
		log_err("config tf failed\n");
		return -1;
	}

	if (is_reset_graph > 0 && tf_reset_graph() != 0) {
		log_warn("tf_reset_graph failed\n");
	}


	if (0 != tf_Session( oConfig)) {
		log_err("create session failed\n");
		return -1;
	}
	log_debug("read model model_type=%d ...\n", model_type);
	if (model_type == 0) {//pb
		if (use_saved_model) {
			log_debug("load_graph_by_saved_model_pb...\n");
			if (0 != load_graph_by_saved_model_pb(signature_key,vec_pb_tags)) {
				return -1;
			}
		}
		else {
			log_debug("tf_load_graph_by_pb...\n");
			PyObject* oRead = NULL;
			/*tf_GFile_read(res, res->m_model_dir.c_str(), &oRead);
			printf("oRead = %lld\n", oRead);*/
			std::string sread;
			if (read_file(m_model_dir.c_str(), sread) <= 0) {
				return -1;
			}
			if (m_aes_conf.use) {
				log_debug("%s aes decode...\n", __FUNCTION__);
				std::string sread_plain;
				if (0 != tk_aes_decode((uint8_t*)sread.c_str(), sread.size(), sread_plain, m_aes_conf.key, m_aes_conf.iv)) {
					log_err("aes decode failed\n");
					return -1;
				}
				oRead = Py_BuildValue("y#", sread_plain.c_str(), sread_plain.size());
			}
			else {
				oRead = Py_BuildValue("y#", sread.c_str(), sread.size());
			}

			if (!oRead) {
				PyErr_Print();
				return -1;
			}
			if (0 != load_graph_by_pb(oRead)) {
				PyErr_Print();
				return -1;
			}
		}
		
	}
	else {//ckpt
		if (0 != tf_load_graph_by_ckpt()) {
			log_err("load ckpt failed\n");
			PyErr_Print();
			return -1;
		}
	}

	if (use_saved_model) {
		if (0 != tf_get_tensor_saved_model()) {
			return -1;
		}
		if (0 != tf_get_tensor()) {
			PyErr_Print();
			return -1;
		}
	}
	else {
		if (0 != tf_get_tensor()) {
			PyErr_Print();
			return -1;
		}
	}

	
	if (!load_sub_func()) {
		log_err("load_sub_func failed\n");
		PyErr_Print();
		return -1;
	}
	return 0;
}












//int tf_GFile_read(C_tf_v1_resource* resource, const char*filename, PyObject**result) {
//
//	auto oTfModule = (*resource)["tensorflow"];
//	PyObject* pFun_gfile = my_PyDict_GetItemString(oTfModule, "gfile");
//	PyObject* pFun_GFile = my_PyDict_GetItemString(pFun_gfile, "GFile");
//
//	PyObject *pthis_GFile = NULL;
//	{
//		auto fname = Py_BuildValue("s", filename);
//		auto mode = Py_BuildValue("s", "rb");
//		PyObject *pArgs = PyTuple_New(2);
//		PyTuple_SetItem(pArgs, 0, fname);
//		PyTuple_SetItem(pArgs, 1, mode);
//
//		auto inst = PyInstanceMethod_New((*resource)["GFile"]);
//		PyObject *pthis_GFile = PyObject_CallObject(inst, pArgs);
//		Py_DECREF(inst);
//		Py_DECREF(pArgs);
//
//
//
//		if (pthis_GFile == NULL) {
//			PyErr_Print();
//			Py_DECREF(pFun_gfile);
//			Py_DECREF(pFun_GFile);
//			Py_DECREF(pthis_GFile);
//			return -1;
//		}
//	}
//
//	{
//		auto func_read = my_PyObject_GetAttrString(pthis_GFile, "read");
//		*result = PyObject_CallObject(func_read, NULL);
//		Py_DECREF(func_read);
//		if (*result == NULL) {
//			PyErr_Print();
//			Py_DECREF(pFun_gfile);
//			Py_DECREF(pFun_GFile);
//			Py_DECREF(pthis_GFile);
//			return -1;
//		}
//	}
//	
//
//	{
//		auto func_close = my_PyObject_GetAttrString(pthis_GFile, "close");
//		PyObject* pArgs3 = PyTuple_New(0);
//		PyObject *result_tmp = PyObject_CallObject(func_close, pArgs3);
//		Py_DECREF(func_close);
//
//		Py_DECREF(pFun_gfile);
//		Py_DECREF(pFun_GFile);
//		if (result_tmp) {
//			Py_DECREF(result_tmp);
//		}
//		else {
//			PyErr_Print();
//		}
//		Py_DECREF(pArgs3);
//		Py_DECREF(pthis_GFile);
//	}
//	
//	
//	return 0;
//	
//}