#ifndef __FT_CSDK_H__
#define __FT_CSDK_H__

#include "../common.h"
#include <map>

class C_ft_resource : public C_engine_base {
public:
	explicit
	C_ft_resource(std::string model_dir, int device_id, std::vector<S_my_net_graph>& net_graph,
		int engine_version, S_aes_option* aes_conf) :
		C_engine_base(model_dir, device_id, net_graph, engine_version, TYPE_ENGINE_TRT, aes_conf) {

		m_handle = NULL;
	}
	virtual ~C_ft_resource() {
		close();
	}
private:

	void close();

public:

	bool OnLoad() {
		return true;
	}

	int OnCreate(int k, float threshold, int predict_label,int dump_label);

	int OnProcess(PyObject** result, int stage,int input_num_, PyObject** inputs_);


	virtual std::vector<std::string> get_labels(){
	    return m_labels;
	}

public:
	void* m_handle;
	std::vector<const void*>m_input_buffer_pt_list;
	std::vector<std::vector<char>>m_input_buffer;
	std::vector<void*>m_output_buf_only_read;
	std::vector<int>m_output_buf_size;

	std::vector<npy_intp>m_output_shape;//当前shape元素 0 batch_size 可能小于配置


	int m_k;
	float m_threshold;
	int m_predict_label;
	int m_dump_label;

	int m_dim{512};

	std::vector<std::string>m_labels;
};

#endif // !__TRT_CSDK_H__
