#ifndef __TF_UTILS_V1_H__
#define __TF_UTILS_V1_H__
#pragma once

#include "../common.h"


class C_tf_v1_resource : public C_engine_base {
public:
	bool m_is_kernel_object;
private:
	std::map<std::string, PyObject*>m_map;
	PyObject* m_ograph, * m_osession, * m_orun;

	std::vector<PyObject*>m_ofetchs;
	PyObject* m_infer_signature;
	PyObject* m_ofeed_dict;

	
public:
	explicit
	C_tf_v1_resource(std::string model_dir, int device_id, std::vector<S_my_net_graph>& net_graph,int engine_version, S_aes_option* aes_conf) :
		C_engine_base(model_dir, device_id, net_graph, engine_version, TYPE_ENGINE_TF2, aes_conf) 
	{
		m_ofeed_dict = NULL;
		m_ograph = NULL;
		m_osession = NULL;
		m_orun = NULL;
		m_ofetchs.clear();
		m_infer_signature = NULL;
		m_is_kernel_object = false;
	}
	virtual ~C_tf_v1_resource() {

		try {
			if (m_osession) {
				auto ret = PyObject_CallMethod(m_osession, "close", NULL);
				if (!ret) {
					//PyErr_Print();
				}
				SAFE_DECREF(ret);
			}

			for (int i = 0; i < m_net_graph.size(); ++i) {
				auto& net_inf_state = m_net_graph[i];
				for (auto& it : net_inf_state.oInput_) {
					SAFE_DECREF(it);
				}
				for (auto& it : net_inf_state.oOutput_) {
					SAFE_DECREF(it);
				}
			}


			SAFE_DECREF(m_ograph);
			SAFE_DECREF(m_osession);


			SAFE_DECREF(m_orun);
			SAFE_DECREF(m_ofeed_dict);


			for (auto& fetch : m_ofetchs) {
				SAFE_DECREF(fetch);
			}
			SAFE_DECREF(m_infer_signature);

			close();
		}
		catch (std::exception& e) {
		}
		
	}

private:
	PyObject* load_attr(PyObject* pmodule, const char* name) {
		if (m_map.find(name) != m_map.end()) {
			return m_map[name];
		}
		PyObject* oo = my_PyObject_GetAttrString(pmodule, name);
		m_map.insert(std::make_pair(name,oo));
		return oo;
	}

	PyObject* load_dict(PyObject* pmodule, const char* name) {
		if (m_map.find(name) != m_map.end()) {
			return m_map[name];
		}
		PyObject* oo = my_PyDict_GetItemString(pmodule, name);
		m_map.insert(std::make_pair(name, oo));
		return oo;
	}

	bool chk_ok() {
		for (auto & x : m_map) {
			if (!x.second) {
				return false;
			}
		}
		return true;
	}
private:
	void close() {
		for (auto & x : m_map) {
			if (x.second) {
				Py_DECREF(x.second);
			}
		}
		m_map.clear();
	}

	int tf_Session(PyObject* oConfig);
	int tf_get_tensor_saved_model();
	int tf_get_tensor();
	int load_graph_by_saved_model_pb(std::string& signature_key, std::vector<std::string>& vec_pb_tags);
	int load_graph_by_pb(PyObject* oRead);
	int tf_load_graph_by_ckpt();
	int tk_ConfigProto_ex(PyObject* oConfigProto_data, PyObject** result);
	int tf_reset_graph();
	
public:

	virtual PyObject* operator[](const char* name)
	{
		if (m_map.find(name) != m_map.end()) {
			return m_map[name];
		}
		return NULL;
	}

	int OnCreate(int model_type, 
		int is_reset_graph,
		PyObject* oConfigProto,
		int  use_fastertransformer, 
		int use_saved_model,
		std::vector<std::string>& vec_pb_tags, std::string& signature_key);

	int OnProcess(PyObject** result, int stage, int input_num_, PyObject** inputs_);
	

	bool OnLoad() {
		Set_cuda_visible_device(m_device_id);
		try {
			PyObject* oModule = NULL;
			oModule = py::module_::import_ex("tensorflow");
			if (!oModule) {
				log_err("%s load tensorflow failed\n", __FUNCTION__);
				PyErr_Print();
				return false;
			}
			
			if (ParseEngineVersion(oModule, m_version, m_ver) == 0) {
			}
			if (m_ver[0] == 1 && m_ver[1] <15) {
				m_is_kernel_object = true;
			}
			if (!m_is_kernel_object) {
				auto oTfDict = PyModule_GetDict(oModule);
				if (!oTfDict) {
					PyErr_Print();
					oTfDict = NULL;
					log_err("%s load tf failed!!\n", __FUNCTION__);
					return false;
				}
				m_map.insert(std::make_pair("tensorflow_base", oModule));
				m_map.insert(std::make_pair("tensorflow", oTfDict));
			}
			else {
				m_map.insert(std::make_pair("tensorflow", oModule));
			}


			//图节点
			for (int i = 0; i < m_net_graph.size(); ++i) {
				auto net_graph = m_net_graph[i];
				net_graph.oInput_.resize(net_graph.input_.size());
				net_graph.oOutput_.reserve(net_graph.output_.size());
			}
		}
		catch (std::exception & e) {
			log_fatal("%s except %s\n", __FUNCTION__, e.what());
			return false;
		}

		return chk_ok();
	}


	bool load_sub_func() {
		log_debug("%s...\n", __FUNCTION__);
		m_ofeed_dict = PyDict_New();

		m_orun = load_attr(m_osession, "run");
		m_ofetchs.resize(m_net_graph.size());
		for (int i = 0; i < m_net_graph.size(); ++i) {
			auto& net_inf = m_net_graph[i];
			auto& ofetch = m_ofetchs[i];
			ofetch = PyTuple_New(net_inf.oOutput_.size());
			for (int j = 0; j < net_inf.oOutput_.size(); ++j) {
				Py_INCREF(net_inf.oOutput_[j]);//输出节点引用均+1
				PyTuple_SetItem(ofetch, j, net_inf.oOutput_[j]);
			}
		}
		return m_orun != NULL;

		/*load_dict(m_map["tensorflow"], "constant");
		load_dict(m_map["tensorflow"], "int64");


		for (int i = 0; i < m_net_graph.size(); ++i) {
			auto & net_inf = m_net_graph[i];
			for (int j = 0; j < net_inf.input_.size(); ++j) {
				auto& dtype = net_inf.input_[j].dtype_long_str;
				if (m_map.find(dtype) != m_map.end()) {
					load_dict(m_map["tensorflow"], dtype.c_str());
				}
			}
		}*/
		return true;
	}
};


#endif // !__PY_UTILS_H__
