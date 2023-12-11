#include "sdk_py.h"

namespace ns_sdk_py {

	int Parse_aes_config(py::dict args, struct S_aes_option& aes_conf) {
		log_debug("%s...\n", __FUNCTION__);
		aes_conf.use = false;
		if (args.contains("aes")) {
			auto ase = args["aes"];
			if (ase.contains("use")) {
				aes_conf.use = py::cast<int>(ase["use"]);
			}
			if (ase.contains("enable")) {
				aes_conf.use = py::cast<int>(ase["enable"]);
			}

			if (aes_conf.use) {
				py::bytes key = py::cast<py::bytes>(ase["key"]);
				py::bytes iv = py::cast<py::bytes>(ase["iv"]);

				auto key_str = key.operator std::string();
				auto iv_str = iv.operator std::string();
				if (key_str.size() != 16 || iv_str.size() != 16) {
					log_err("%s the dim of aes key or iv must 16\n", __FUNCTION__);
					return -1;
				}
				memcpy(aes_conf.key, key_str.c_str(), 16);
				memcpy(aes_conf.iv, iv_str.c_str(), 16);
			}
		}
		return 0;
	}
	
	
	int Parse_tf_config(py::dict args, 
		int& engine_version,
		int& is_reset_graph,
		PyObject*& oConfigProto,
		int& model_type,
		int& use_fastertransformer,
		int& use_saved_model,
		std::string& signature_key,
		std::vector<std::string>& vec_pb_tags) {

		log_debug("%s...\n", __FUNCTION__);
		auto tf = args["tf"];
		if (!tf) {
			log_err("%s config missing tf\n", __FUNCTION__);
			return -1;
		}
	
		if (tf.contains("engine_major")) {
			engine_version = py::cast<int>(tf["engine_major"]);
		}

		if (tf.contains("is_reset_graph")) {
			is_reset_graph = py::cast<int>(tf["is_reset_graph"]);
		}
		
		if (tf.contains("engine_version")) {
			engine_version = py::cast<int>(tf["engine_version"]);
		}
		if (engine_version != 1 && engine_version != 2) {
			log_err("%s config.tf engine_version %d , not in [1,2]\n", __FUNCTION__, engine_version);
			return -1;
		}

		if (tf.contains("ConfigProto")) {
			oConfigProto = tf["ConfigProto"].ptr();
		}
		

		model_type = py::cast<int>(tf["model_type"]);
		if (model_type != 0 && model_type != 1) {
			log_err("%s model_type not in [0,1]\n", __FUNCTION__);
			return -1;
		}

		if (tf.contains("fastertransformer")) {
			auto ft = tf["fastertransformer"];
			if (ft.contains("enable")) {
				use_fastertransformer = py::cast<int>(ft["enable"]);
			}
			if (ft.contains("use")) {
				use_fastertransformer = py::cast<int>(ft["use"]);
			}
		}
		if (tf.contains("saved_model")) {
			auto saved_model = tf["saved_model"];
			if (saved_model.contains("use")) {
				use_saved_model = py::cast<int>(saved_model["use"]);
			}

			if (saved_model.contains("enable")) {
				use_saved_model = py::cast<int>(saved_model["enable"]);
			}

			if (use_saved_model) {
				if (saved_model.contains("signature_key")) {
					signature_key = py::str(saved_model["signature_key"]).operator std::string();
				}
				if (saved_model.contains("tags")) {
					py::list tags = py::cast<py::list>(saved_model["tags"]);
					auto nsize = tags.size();
					for (auto i = 0; i < nsize; ++i) {
						vec_pb_tags.push_back(py::str(tags[i]).operator std::string());
					}
				}
				log_info("saved_model tags size %d\n", vec_pb_tags.size());
				for (auto& it : vec_pb_tags) {
					log_info("tags: %s\n", it.c_str());
				}
			}
		}
		return 0;
	}
	
	
	int Parse_onnx_config(py::dict args, int& engine_version,int& enable_tensorrt) {
		log_debug("%s...\n", __FUNCTION__);
		engine_version = 1;
		enable_tensorrt = 1;
		if (args.contains("onnx")) {
			auto onnx = args["onnx"];

			if (onnx.contains("engine_major")) {
				engine_version = py::cast<int>(onnx["engine_major"]);
			}

			if (onnx.contains("engine_version")) {
				engine_version = py::cast<int>(onnx["engine_version"]);
			}

			if (onnx.contains("tensorrt")) {
				enable_tensorrt = py::cast<int>(onnx["tensorrt"]);
			}

		}
		return 0;
	}
	
	
	
	int Parse_trt_config(py::dict args, int& engine_version,int engine_minor, int& enable_graph) {
		log_debug("%s...\n", __FUNCTION__);
		engine_version = 8;
		engine_minor = 0;
		if (args.contains("trt")) {
			auto trt = args["trt"];

			if (trt.contains("engine_major")) {
				engine_version = py::cast<int>(trt["engine_major"]);
			}


			if (trt.contains("engine_minor")) {
				engine_minor = py::cast<int>(trt["engine_minor"]);
			}
			if (engine_version != 7 && engine_version != 8) {
				log_err("%s tensorrt engine_version only support 7 8\n", __FUNCTION__);
				return -1;
			}
			if (trt.contains("enable_graph")) {
				enable_graph = py::cast<int>(trt["enable_graph"]);
			}
		}
		return 0;
	}

	int Parse_fasttext_config(py::dict args, int& engine_version,int & k,float & threshold,int & predict_label,int & dump_label) {
		log_debug("%s...\n", __FUNCTION__);
		engine_version = 8;
		if (args.contains("fasttext")) {
			auto ft = args["fasttext"];

			if (ft.contains("engine_major")) {
				engine_version = py::cast<int>(ft["engine_major"]);
			}

			if (ft.contains("engine_version")) {
				engine_version = py::cast<int>(ft["engine_version"]);
			}
			if (ft.contains("threshold")) {
				threshold = py::cast<float>(ft["threshold"]);
			}
			if (ft.contains("k")) {
				k = py::cast<int>(ft["k"]);
			}

			if (ft.contains("predict_label")) {
				predict_label = py::cast<int>(ft["predict_label"]);
			}
			if (ft.contains("dump_label")) {
				dump_label = py::cast<int>(ft["dump_label"]);
			}
		}
		return 0;
	}
	
	
	int Parse_graph_config(py::dict args, std::vector<S_my_net_graph>& net_inf_graph) {
		log_debug("%s...\n", __FUNCTION__);
		if (!args.contains("graph")) {
			log_err("%s config missing graph\n",__FUNCTION__);
			return -1;
		}
		py::list net_graph = args["graph"];
		int net_size = net_graph.size();
		net_inf_graph.resize(net_size);
		bool berr = false;
		auto ParseInput_Output = [](S_my_net_graph& cur_net_inf_graph, py::list sub_i_o_graph, bool is_input) {
			S_my_graph_node tmp_node;

			int input_ouput_size = sub_i_o_graph.size();
			if (!is_input && !input_ouput_size) {
				log_err("Parse_graph_config graph input ,output config is not right\n");
				return -1;
			}
			for (int j = 0; j < input_ouput_size; ++j) {
				tmp_node.shape.clear();
				auto node = sub_i_o_graph[j];

				tmp_node.name = py::str(node["node"]).operator std::string();
				std::string str_data_type = "int64";
				if (node.contains("dtype")) {
					str_data_type = py::str(node["dtype"]).operator std::string();
				}else if (node.contains("data_type")) {
					str_data_type = py::str(node["data_type"]).operator std::string();
				}
				tmp_node.data_type = Get_dtype_from_string(str_data_type.c_str());
				Get_dtype_string(tmp_node.data_type, tmp_node.dtype_short_str, tmp_node.dtype_long_str);

				if (node.contains("shape")) {
					py::list shape = node["shape"];
					for (Py_ssize_t kk = 0, N_size = shape.size(); kk < N_size; ++kk) {
						tmp_node.shape.push_back(shape[kk].is_none()? -1: py::cast<int>(shape[kk]));
					}
				}

				std::string shape_str = "(";
				for (int m = 0; m < tmp_node.shape.size(); ++m) {
					shape_str += std::to_string(tmp_node.shape[m]);
					if (m != tmp_node.shape.size() - 1) {
						shape_str += ",";
					}
				}
				shape_str += ")";

				if (is_input) {
					log_info("input %d node %s , %s , shape: %s\n", j, tmp_node.name.c_str(), tmp_node.dtype_long_str.c_str(), shape_str.c_str());
					cur_net_inf_graph.input_.push_back(tmp_node);
				}
				else {
					log_info("output %d node %s , %s , shape: %s\n", j, tmp_node.name.c_str(), tmp_node.dtype_long_str.c_str(), shape_str.c_str());
					cur_net_inf_graph.output_.push_back(tmp_node);
				}
			}
			if (is_input) {
				cur_net_inf_graph.oInput_.resize(input_ouput_size);
			}
			else {
				cur_net_inf_graph.oOutput_.resize(input_ouput_size);
			}
			return 0;

		};
		log_info("parsing graph info...\n");
		for (int i = 0; i < net_size; ++i) {
			log_info("sub_graph %d:\n", i);
			auto& tmp_net = net_inf_graph[i];
			py::dict sub_net_graph = net_graph[i];

			py::list sub_input_graph = sub_net_graph["input"];
			py::list sub_output_graph = sub_net_graph["output"];

			if (0 != ParseInput_Output(tmp_net, sub_input_graph, true)) {
				berr = true;
				break;
			}
			if (0 != ParseInput_Output(tmp_net, sub_output_graph, false)) {
				berr = true;
				break;
			}
		}
		if (berr) {
			return -1;
		}
		return 0;
	}
}