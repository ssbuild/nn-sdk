#ifndef __ONNX_CSDK_H__
#define __ONNX_CSDK_H__
#include "../common.h"

class C_onnx_resource : public C_engine_base {
public:
	std::map<std::string, PyObject*>m_map;
	PyObject* m_ograph, * m_osession, * m_orun;
	std::vector<PyObject*>m_ofetchs;

	PyObject* m_ofeed_dict;

	static py::object ms_onnxruntime;
	static py::object ms_onnxruntime_dict;
public:
	explicit
	C_onnx_resource(std::string model_dir, int device_id, std::vector<S_my_net_graph>& net_graph,int engine_version,
		S_aes_option* aes_conf) :
		C_engine_base(model_dir, device_id, net_graph, engine_version, TYPE_ENGINE_ONNX, aes_conf)
	{

		m_ograph = NULL;
		m_osession = NULL;
		m_orun = NULL;
		m_ofeed_dict = NULL;
		m_ofetchs.clear();
		
	}
	virtual ~C_onnx_resource() {

		try {
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

			close();
		}
		catch (std::exception & e) {
		}
		
	}

private:
	PyObject* load_attr(PyObject* pmodule, const char* name) {
		if (m_map.find(name) != m_map.end()) {
			return m_map[name];
		}
		PyObject* oo = my_PyObject_GetAttrString(pmodule, name);
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
public:

	int OnCreate(int enable_tensorrt);

	int OnProcess(PyObject** result, int stage, int input_num_, PyObject** inputs_);
private:
	void close() {
		for (auto & x : m_map) {
			if (x.second == ms_onnxruntime.ptr()) {
				continue;
			}
			if (x.second == ms_onnxruntime_dict.ptr()) {
				continue;
			}

			
			if (x.second) {
				Py_DECREF(x.second);
			}
		}
		m_map.clear();
	}

	int onnx_Session(PyObject* oRead);

	int onnx_Session_ex(PyObject* oRead, int enable_tensorrt);

	int onnx_make_feeds_fetch();
public:

	virtual PyObject* operator[](const char* name)
	{
		if (m_map.find(name) != m_map.end()) {
			return m_map[name];
		}
		return NULL;
	}

	bool OnLoad() {
		log_debug("%s...\n", __FUNCTION__);
		try {

			PyObject* oModule = NULL;
			if (ms_onnxruntime.is_none()) {
				oModule = py::module_::import_ex("onnxruntime");
				ms_onnxruntime = py::reinterpret_borrow<py::object>(py::handle(oModule));
			}
			else {
				oModule = ms_onnxruntime.ptr();
			}
		
			if (!oModule) {
				PyErr_Print();
				log_err("%s load onnxruntime failed\n", __FUNCTION__);
				return false;
			}

			if (ParseEngineVersion(oModule, m_version, m_ver) == 0) {
				
			}


			//低版本设置卡
			if (m_ver[0] <= 1 && m_ver[1]<=4) {
				Set_cuda_visible_device(m_device_id);
			}
			m_map.insert(std::make_pair("onnxruntime_base", oModule));

			PyObject* oOnnxruntimeDict = NULL;
			if (ms_onnxruntime_dict.is_none()) {
				oOnnxruntimeDict = PyModule_GetDict(oModule);
				ms_onnxruntime_dict = py::reinterpret_borrow<py::dict>(py::handle(oOnnxruntimeDict));
			}
			else {
				oOnnxruntimeDict = ms_onnxruntime_dict.ptr();
			}
			m_map.insert(std::make_pair("onnxruntime", oOnnxruntimeDict));
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
			auto & net_inf = m_net_graph[i];
			auto & ofetch = m_ofetchs[i];
			ofetch = PyList_New(net_inf.oOutput_.size());
			for (int j = 0; j < net_inf.oOutput_.size(); ++j) {
				Py_INCREF(net_inf.oOutput_[j]);//输出节点引用均+1
				PyList_SetItem(ofetch, j, net_inf.oOutput_[j]);
			}
		}
		return m_orun != NULL;
	}


};

#endif
