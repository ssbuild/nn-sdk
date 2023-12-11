#ifndef __TRT_CSDK_H__
#define __TRT_CSDK_H__

#include "../common.h"

class C_trt_resource : public C_engine_base {
public:
	explicit
	C_trt_resource(std::string model_dir, int device_id, std::vector<S_my_net_graph>& net_graph,
		int engine_version, int engine_minor, S_aes_option* aes_conf) :
		C_engine_base(model_dir, device_id, net_graph, engine_version,TYPE_ENGINE_TRT, aes_conf) {

		m_trt_handle = NULL;
		m_engine_minor = engine_minor;
	}
	virtual ~C_trt_resource() {
		close();
	}
private:

	void close();

public:

	bool OnLoad() {
		return true;
	}

	int OnCreate(int enable_graph);

	int OnProcess(PyObject** result, int stage,int input_num_, PyObject** inputs_);

public:
	void* m_trt_handle;
	std::vector<const void*>m_input_buffer_pt_list;
	std::vector<std::vector<char>>m_input_buffer;
	std::vector<void*>m_output_buf_only_read;
	std::vector<int>m_output_buf_size;

	std::vector<npy_intp>m_output_shape;//当前shape元素 0 batch_size 可能小于配置
	int m_engine_minor{ 0 };
};

#endif // !__TRT_CSDK_H__
