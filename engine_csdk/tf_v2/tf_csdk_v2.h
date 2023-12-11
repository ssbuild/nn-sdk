#ifndef __TF_UTILS_V2_H__
#define __TF_UTILS_V2_H__
#pragma once

#include "../common.h"


class C_tf_v2_resource : public C_engine_base {
private:
	PyObject* m_ofeed_dict, *m_orun_args_0;
public:
	explicit
	C_tf_v2_resource(std::string model_dir, int device_id, std::vector<S_my_net_graph>& net_graph, 
		int engine_version, S_aes_option* aes_conf) :
		C_engine_base(model_dir, device_id, net_graph, engine_version, TYPE_ENGINE_TF2, aes_conf) {

		m_ograph = NULL;
		m_osession = NULL;
		m_orun = NULL;
		m_ofetchs.clear();
		m_infer_func = NULL;
		m_ofeed_dict = NULL;
		m_orun_args_0 = NULL;
		m_is_kernel_object = false;

	}
	virtual ~C_tf_v2_resource() {

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
			SAFE_DECREF(m_orun_args_0);



			for (auto& fetch : m_ofetchs) {
				SAFE_DECREF(fetch);
			}
			SAFE_DECREF(m_infer_func);


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

	int load_graph_by_saved_model_pb_v2(std::string& signature_key, std::vector<std::string>& vec_pb_tags);
public:

	virtual PyObject* operator[](const char* name)
	{
		if (m_map.find(name) != m_map.end()) {
			return m_map[name];
		}
		return NULL;
	}

	
	int OnCreate(int model_type,int use_saved_model,std::vector<std::string>& vec_pb_tags, std::string& signature_key);

	//仿tf1 , 限制子图的输入输出
	int OnProcess(PyObject** result, int stage, int input_num_, PyObject** inputs_);



	bool OnLoad(PyObject* oConfigProto) {
		Set_cuda_visible_device(m_device_id);
		try {
			PyObject* oModule = NULL;
			oModule = py::module_::import_ex("tensorflow");

			/*auto m = py::module_::import("tensorflow");
			oModule = m.ptr();*/

			if (!oModule) {
				log_err("%s load tensorflow failed\n", __FUNCTION__);
				PyErr_Print();
				return false;
			}
			
			if (ParseEngineVersion(oModule, m_version, m_ver) == 0) {
			}

			if (m_ver[0] == 1 && m_ver[1] < 15) {
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


			if(oConfigProto != NULL && m_device_id >=0)
			{

				auto py_oconfig = py::reinterpret_steal<py::dict>(oConfigProto);
				long global_jit_level = 0;
				if (py_oconfig.contains("graph_options")) {
					if (py_oconfig["graph_options"].contains("optimizer_options")) {
						auto py_optimizer_options = py_oconfig["graph_options"]["optimizer_options"];
						if (py_optimizer_options) {
							global_jit_level = py::cast<long>(py_optimizer_options["global_jit_level"]);
						}
						
					}
				}
			
				auto allow_soft_placement = PyDict_GetItem(oConfigProto,PyUnicode_FromString("allow_soft_placement"));
				auto gpu_options = PyDict_GetItem(oConfigProto, PyUnicode_FromString("gpu_options"));

				/*PyObject_Print(allow_soft_placement, stdout, 0);
				PyObject_Print(gpu_options, stdout, 0);*/

				PyObject* allow_growth = NULL;
				if (gpu_options) {
					allow_growth = PyDict_GetItem(gpu_options, PyUnicode_FromString("allow_growth"));
				}
				py::module_ tf_module = py::reinterpret_steal<py::module_>(m_map["tensorflow"]);
				if (tf_module.contains("config")) {
					py::module_ oconfig = tf_module["config"];

					if (global_jit_level) {
						auto ooptimizer = oconfig.attr("optimizer");  
						if (ooptimizer) {
							auto oset_jit = ooptimizer.attr("set_jit");
							if (oset_jit) {
								log_debug("set_jit %d\n", global_jit_level);
								oset_jit(global_jit_level);
							}
						}
					}


					if (allow_soft_placement) {
						auto oset_soft_device_placement = oconfig.attr("set_soft_device_placement");
						if (oset_soft_device_placement) {
							log_debug("set_soft_device_placement %d\n", PyLong_AsLong(allow_soft_placement));
							oset_soft_device_placement(PyLong_AsLong(allow_soft_placement));
						}
					}
					if (allow_growth) {
						auto olist_physical_devices = oconfig.attr("list_physical_devices");
						if (olist_physical_devices) {
							auto ophysical_devices = py::cast<py::list>( olist_physical_devices("GPU"));
							int nGpu = ophysical_devices.size();
							if (nGpu) {
								auto first_gpu = ophysical_devices[0];
								auto oexperimental = oconfig.attr("experimental");
								if (oexperimental) {
									auto oset_memory_growth = oexperimental.attr("set_memory_growth");
									if (oset_memory_growth) {
										oset_memory_growth(first_gpu,PyLong_AsLong(allow_growth));
									}
								}
							}
						}
					}
				}
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
		m_orun_args_0 = PyTuple_New(0);
		load_dict(m_map["tensorflow"], "constant");
		load_dict(m_map["tensorflow"], "int64");
		load_dict(m_map["tensorflow"], "int32");
		load_dict(m_map["tensorflow"], "float32");


		m_ofetchs.resize(m_net_graph.size());
		for (int i = 0; i < m_net_graph.size(); ++i) {
			auto & net_inf = m_net_graph[i];
			for (int j = 0; j < net_inf.input_.size(); ++j) {
				auto& dtype = net_inf.input_[j].dtype_long_str;
				if (m_map.find(dtype) == m_map.end()) {
					load_dict(m_map["tensorflow"], dtype.c_str());
				}
			}
		}
		return true;
	}

	std::map<std::string, PyObject*>m_map;
	PyObject *m_ograph, *m_osession, *m_orun;


	std::vector<PyObject*>m_ofetchs;

	PyObject *m_infer_func;
	bool m_is_kernel_object;
	std::vector<PyObject*>m_vec_inputs;

};





#endif // !__PY_UTILS_H__
