#include "tf_v1/tf_csdk_v1.h"
#include "tf_v2/tf_csdk_v2.h"
#include "onnx/onnx_csdk.h"
#include "trt/trt_csdk.h"
#include "ft/ft_csdk.h"
#include "sdk_py.h"

namespace ns_sdk_py {


	py::object sdk_new(py::dict args) {
		log_info("nn-sdk version: %s\n", version::get_version().c_str());
		if (!Py_IsInitialized()) {
			Py_Initialize();
		}
		int iret = 0;

		the_config.log_level = LOG_LEVEL_INFO;
		std::string model_dir;
		long long resouce_id = 0;
		struct S_aes_option aes_conf;
		int engine_type = 0;
		int device_id = 0, model_type = 0, engine_major = 1, engine_minor = 0, enable_tensorrt =0;
		std::vector<S_my_net_graph> net_inf_graph;
		do {
			PyObject* oConfigProto = NULL;
			model_dir = py::str(args["model_dir"]).operator std::string();
			if (args.contains("log_level")) {
				the_config.log_level = py::cast<int>(args["log_level"]);
			}
			if (args.contains("engine")) {
				engine_type = py::cast<int>(args["engine"]);
			}
			if (engine_type < 0 && engine_type >=4) {
				log_err("engine must be [0-3]\n");
				break;
			}
			if (args.contains("device_id")) {
				device_id = py::cast<int>(args["device_id"]);
			}
			
			iret = Parse_aes_config(args, aes_conf);
			if (iret != 0) break;

			iret = Parse_graph_config(args, net_inf_graph);
			if (iret != 0) break;

			if (engine_type == 0) {
				int is_reset_graph = 1;
				int  use_fastertransformer = 0;
				int use_saved_model = 0;
				std::vector<std::string>vec_pb_tags;
				std::string signature_key = "serving_default";

				iret = Parse_tf_config(args,  
					engine_major, 
					is_reset_graph,
					oConfigProto,
					model_type,
					use_fastertransformer,
					use_saved_model,
					signature_key,
					vec_pb_tags);

				if (iret != 0) break;

				log_info("model_dir %s,engine %d,engine_major %d,device_id=%d,ase.use=%d\n",
					model_dir.c_str(), engine_type, engine_major, device_id,
					aes_conf.use);

				if (engine_major == 1) {

					C_tf_v1_resource* inst = new C_tf_v1_resource(model_dir, device_id, net_inf_graph, engine_major, &aes_conf);
					if (inst) {
						if (!inst->OnLoad()) {
							delete inst;
						}
						else {
							if (0 == inst->OnCreate(model_type, is_reset_graph,
									oConfigProto, use_fastertransformer,  
									use_saved_model, vec_pb_tags, signature_key)) {
								resouce_id = (long long)inst;
							}
							else {
								delete inst;
							}
						}
					}
				}
				else if (engine_major == 2) {
					
					C_tf_v2_resource* inst = new C_tf_v2_resource(model_dir, device_id, net_inf_graph, engine_major, &aes_conf);
					if (inst) {
						if (!inst->OnLoad(oConfigProto)) {
							delete inst;
						}
						else {
							if (use_saved_model == 0) {
								use_saved_model = 1;
							}
							if (0 == inst->OnCreate(model_type, use_saved_model, vec_pb_tags, signature_key)) {
								resouce_id = (long long)inst;
							}
							else {
								delete inst;
							}

						}
					}
				}
				
			}
			else if (engine_type == 1) {
				iret = Parse_onnx_config(args, engine_major, enable_tensorrt);
				if (iret != 0) break;

				log_info("model_dir %s,engine %d,engine_major %d,device_id=%d,ase.use=%d\n",
					model_dir.c_str(), engine_type, engine_major, device_id,
					aes_conf.use);
				C_onnx_resource* inst = new C_onnx_resource(model_dir, device_id, net_inf_graph, engine_major, &aes_conf);
				if (inst) {
					if (!inst->OnLoad()) {
						delete inst;
					}
					else {
						if (0 == inst->OnCreate(enable_tensorrt)) {
							resouce_id = (long long)inst;
						}
						else {
							delete inst;
						}
					}
				}
			}
			else if (engine_type == 2) {
				int enable_graph = 0;
				iret = Parse_trt_config(args, engine_major, engine_minor,enable_graph);
				if (iret != 0) break;
				log_info("model_dir %s,engine %d,engine_major %d,device_id=%d,ase.use=%d\n",
					model_dir.c_str(), engine_type, engine_major, device_id,
					aes_conf.use);
				C_trt_resource* inst = new C_trt_resource(model_dir, device_id, net_inf_graph, engine_major, engine_minor, &aes_conf);
				if (inst) {
					if (!inst->OnLoad()) {
						delete inst;
					}
					else {
						if (0 == inst->OnCreate(enable_graph)) {
							resouce_id = (long long)inst;
						}
						else {
							delete inst;
						}
					}
				}
			}
			else if (engine_type == 3) {
				int k = 1;
				float threshold = 0;
				int predict_label = 0;
				int dump_label = 1;
				iret = Parse_fasttext_config(args, engine_major,k, threshold, predict_label, dump_label);
				if (iret != 0) break;
				log_info("model_dir %s,engine %d,engine_major %d,device_id=%d,ase.use=%d\n",
					model_dir.c_str(), engine_type, engine_major, device_id,
					aes_conf.use);
				C_ft_resource* inst = new C_ft_resource(model_dir, device_id, net_inf_graph, engine_major, &aes_conf);
				if (inst) {
					if (!inst->OnLoad()) {
						delete inst;
					}
					else {
						if (0 == inst->OnCreate(k, threshold, predict_label, dump_label)) {
							resouce_id = (long long)inst;
						}
						else {
							delete inst;
						}
					}
				}
			}
			else {
				log_err("engine_type %d no support\n", engine_type);
			}
		} while (0);
		log_debug("%s %lld\n", __FUNCTION__, resouce_id);
		return py::cast<py::object>(Py_BuildValue("L", resouce_id));
	}

	py::object sdk_process(py::args args) {

		long long resouce_id = 0;
		int code = -1;

		py::tuple ret_tuple(2);

		PyObject* result = NULL;
		int input_num = args.size();
		input_num -= 2;

		bool handle = false;
		if (input_num >= 0) {
			resouce_id = py::cast<long long>(args[0]);
			if (resouce_id > 0) {
				handle = true;
				auto engine = ((C_engine_base*)resouce_id);
				auto& input = engine->m_lts_input;
				input.resize(input_num);
				long stage = py::cast<long>(args[1]);
				for (int i = 0; i < input_num; ++i) {
					input[i] = args[i + 2].ptr();
				}
				code = engine->OnProcess(&result, stage, input_num, input.data());
			}
		}

		ret_tuple[0] = py::int_(code);
		if (code != 0) {
			ret_tuple[1] = py::none();
		}
		else {
			ret_tuple[1] = py::reinterpret_steal<py::object>(result);
		}
		return ret_tuple;
	}


	int sdk_delete(py::object args) {
		long long resouce_id = 0;
		int code = -1;
		do {
			resouce_id = py::cast<long long>(args);
			if (resouce_id) {
				code = 0;
				auto engine = ((C_engine_base*)resouce_id);
				delete engine;
			}
		} while (0);

		return  code;
	}

	py::str sdk_version() {
		auto ver = version::get_build_time();
		return py::str(ver);
	}


	py::list sdk_labels(py::object args) {
		long long resouce_id = 0;
		int code = -1;
		do {
			resouce_id = py::cast<long long>(args);
			if (resouce_id) {
				code = 0;
				auto engine = ((C_engine_base*)resouce_id);
				auto labels = engine->get_labels();
				py::list dlist;
				for(auto label : labels){
				    dlist.append(py::str(label));
				}
				return dlist;
			}
		} while (0);

		return  py::none();
	}


	int sdk_init() {
		int code = 0;
		if (!Py_IsInitialized()) {
			Py_Initialize();
		}
		return code;
	}

	int sdk_uninit() {

		Py_Finalize();
		return 0;
	}



	py::tuple sdk_aes_encode_decode(py::dict param) {
		int code = -1;
		py::bytes ret = py::none();
		do {
			int mode = py::cast<int>(param["mode"]);
			PyBytesObject* data = (PyBytesObject*)param["data"].ptr();
			PyBytesObject* key = (PyBytesObject*)param["key"].ptr();
			PyBytesObject* iv = (PyBytesObject*)param["iv"].ptr();
			if (Py_SIZE(key) != 16 || Py_SIZE(iv) != 16) {
				log_err("%s key and iv size must 16!\n", __FUNCTION__);
				break;
			}
			auto key_bytes = key->ob_sval;
			auto iv_bytes = iv->ob_sval;

			int data_size = Py_SIZE(data);
			auto data_bytes = data->ob_sval;

			std::string output;
			if (mode == 0) {//加密
				code = tk_aes_encode((uint8_t*)data_bytes, data_size, output, (uint8_t*)key, (uint8_t*)iv);
			}
			else {
				code = tk_aes_decode((uint8_t*)data_bytes, data_size, output, (uint8_t*)key, (uint8_t*)iv);
			}
			if (code == 0) {
				ret = py::bytes(output);
			}
		} while (0);

		py::tuple result(2);
		result[0] = py::int_(code);
		result[1] = ret;
		return result;
	}

#if NN_DEBUG
	py::object __test__(py::args& args) {

		printf("内部 args %d\n", args.ptr()->ob_refcnt);
		py::tuple result(args.size());

		for (int i = 0; i < args.size(); ++i) {
			result[i] = args[i];
			//Py_DECREF(args[i].ptr());
		}

		/*{
			py::dict param;
			printf("%s %d\n", __FUNCTION__, param.ptr()->ob_refcnt);
			param["0"] = py::str("中国人as打算多阿萨德ad阿萨德as啊是发送到发送到发送到封是的");
			param["1"] = py::str("中国人as打算多阿萨德ad阿萨德a2342342343242342342334211s啊");

			printf("param %d\n", param.ptr()->ob_refcnt);

			printf("param  0 %d  \n", param["0"].ptr()->ob_refcnt);
			printf("param  1 %d  \n", param["1"].ptr()->ob_refcnt);
		}*/


		return result;
	}
#endif

}

PYBIND11_MODULE(engine_csdk, m) {

	m.doc() = "engine_csdk module";
	
	m.def("sdk_init", ns_sdk_py::sdk_init);
	m.def("sdk_uninit", ns_sdk_py::sdk_uninit);
	m.def("sdk_new", ns_sdk_py::sdk_new);
	m.def("sdk_delete", ns_sdk_py::sdk_delete);
	m.def("sdk_process", ns_sdk_py::sdk_process);
	m.def("sdk_version", ns_sdk_py::sdk_version);
	m.def("sdk_labels", ns_sdk_py::sdk_labels);
	m.def("sdk_aes_encode_decode", ns_sdk_py::sdk_aes_encode_decode);
#if NN_DEBUG
	m.def("__test__", __test__);
#endif // NN_DEBUG

	m.attr("__version__") = py::str(version::get_version());
	// Add bindings here
	/*m.def("foo", []() {return "Hello, World!";});
	m.def("add", [](int a, int b) {return a + b;});
	m.def("sub", [](int a, int b) {return a - b;});
	m.def("mul", [](int a, int b) {return a * b;});
	m.def("div", [](int a, int b) {return static_cast<float>(a) / b;});*/

}


