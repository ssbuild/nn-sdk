#ifndef __SDK_PY_H__
#define __SDK_PY_H__
#pragma once
#include "common.h"


namespace ns_sdk_py {


	int Parse_aes_config(py::dict args, struct S_aes_option& aes_conf);

	int Parse_tf_config(py::dict args, 
		int& engine_version, 
		int & is_reset_graph,
		PyObject*& oConfigProto,
		int& model_type,
		int& use_fastertransformer,
		int& use_saved_model,
		std::string& signature_key,
		std::vector<std::string>& vec_pb_tags);
	
	int Parse_onnx_config(py::dict args, int& engine_version,int & enable_tensorrt);

	int Parse_trt_config(py::dict args, int& engine_version,int engine_minor, int& enable_graph);

	int Parse_fasttext_config(py::dict args, int& engine_version, int& k, float& threshold, int& predict_label,int & dump_label);

	int Parse_graph_config(py::dict args, std::vector<S_my_net_graph>&net_inf_graph);


	py::object sdk_new(py::dict args);

	py::object sdk_process(py::args args);

	int sdk_delete(py::object args);

	py::list sdk_labels(py::object args);

	py::str sdk_version();

	int sdk_init();

	int sdk_uninit();

	py::tuple sdk_aes_encode_decode(py::dict param);
}

#endif // !__SDK_PY_H__