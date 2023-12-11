#ifndef __COMMON_H__
#define __COMMON_H__


#include <regex>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <map>
#include <algorithm>
#include <numeric>
#include "version.h"
#include "aes.h"

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"

#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"


namespace  py = pybind11;

#if _WIN32
#else
#define TCHAR
#endif

#define SAFE_DECREF(x) if(x){Py_DECREF(x); x = NULL;}

bool my_PyDict_HasItemString(PyObject* dict, const char*k);
PyObject* my_PyDict_GetItemString(PyObject* dict, const char*k);
PyObject* my_PyObject_GetAttrString(PyObject* dict, const char*k);
PyObject* my_PyObject_GetItemString(PyObject* dict, const char*k);

int Set_cuda_visible_device(int device_id);

int ParseEngineVersion(PyObject * oengine_module, std::string & version, int ver[2]);

void My_splict_string(std::string src, const char* splict, std::vector<std::string>& vec_string);

int Get_dtype_from_string(const char* str_data_type);

void Get_dtype_string(int data_type,std::string & dtype_short, std::string & dtype_long);

int Get_dsize_by_type(int np_type);

void GetList_to_buffer(PyObject * oList, char*& buffer);


enum E_log_level
{
	LOG_LEVEL_DIRECT = -1,
	LOG_LEVEL_FATAL = 0,
	LOG_LEVEL_ERR = 2,
	LOG_LEVEL_WARN = 4,
	LOG_LEVEL_INFO = 8,
	LOG_LEVEL_DEUBG = 16,
};
void LOG_V(E_log_level level, const char* format, va_list& args);
void log_direct(const char* format, ...);
void log_fatal(const char* format, ...);
void log_err(const char* format, ...);
void log_warn(const char* format, ...);
void log_info(const char* format, ...);
void log_debug(const char* format, ...);



int read_file(const char*filename, std::string &content);

struct S_my_graph_node {
	S_my_graph_node() {}
	S_my_graph_node(const S_my_graph_node & other) {
		name = other.name;
		data_type = other.data_type;
		shape = other.shape;
		dtype_long_str = other.dtype_long_str;
		dtype_short_str = other.dtype_short_str;

	}
	std::string name;
	int data_type;
	std::string dtype_long_str;
	std::string dtype_short_str;
	std::vector<npy_intp>shape;
};

class S_my_net_graph {
public:
	S_my_net_graph() {
	}
	S_my_net_graph(const S_my_net_graph & other) {
		input_ = other.input_;
		oInput_ = other.oInput_;
		output_ = other.output_;
		oOutput_ = other.oOutput_;
	}
	~S_my_net_graph() {
	}
	std::vector<S_my_graph_node>input_;
	std::vector<PyObject*>oInput_;

	std::vector<S_my_graph_node>output_;
	std::vector<PyObject*>oOutput_;
};

typedef struct S_user_config {
	int log_level{ LOG_LEVEL_DEUBG };
}*PS_user_config;

typedef struct S_aes_option {
	bool use{0};
	unsigned char key[16];
	unsigned char iv[16];
}*PS_aes_option;

extern struct S_user_config the_config;

enum E_engine_type {
	TYPE_ENGINE_TF1 = 0,
	TYPE_ENGINE_TF2 = 1,
	TYPE_ENGINE_ONNX = 2,
	TYPE_ENGINE_TRT = 3,
};

class C_engine_base {
public:
	int m_engine_major;
	std::vector<S_my_net_graph>m_net_graph;
	E_engine_type m_engine_type;
	std::string m_model_dir, m_version;
	int m_device_id;
	S_aes_option m_aes_conf;
	std::vector<PyObject*>m_lts_input;
	int m_ver[2];
public:
	explicit
	C_engine_base(std::string& model_dir ,int device_id, std::vector<S_my_net_graph>& net_graph,
		int engine_major,int engine_type, S_aes_option* aes_conf) {
		m_model_dir = model_dir;
		m_engine_major = engine_major;
		m_net_graph = net_graph;
		m_engine_type = (E_engine_type)engine_type;
		m_device_id = device_id;
		if (aes_conf) {
			memcpy(&m_aes_conf, aes_conf, sizeof(S_aes_option));
		}
		else {
			m_aes_conf.use = false;
		}

		if (m_engine_type == TYPE_ENGINE_TF2) {
			m_ver[0] = 2;
			m_ver[1] = 5;
		}
		else if (m_engine_type == TYPE_ENGINE_ONNX) {
			m_ver[0] = 1;
			m_ver[1] = 8;
		}
		else if (m_engine_type == TYPE_ENGINE_TRT) {
			m_ver[0] = 8;
			m_ver[1] = 0;
		}else{
			m_ver[0] = 1;
			m_ver[1] = 15;
		}
	}
	virtual ~C_engine_base() {};

	virtual int get_engine_type() {
		return m_engine_type;
	}
	virtual int get_engine_version() {
		return m_engine_major;
	}

	virtual int OnProcess(PyObject** result, int stage, int input_num_, PyObject** inputs_) = 0;

	/*virtual bool OnLoad() = 0;*/

	virtual std::vector<std::string> get_labels(){
	    std::vector<std::string>labels;
	    return labels;
	}

};

#ifdef _WIN32
#define mystrcmp strcmpi
#else
#define mystrcmp strcasecmp
#endif // _win32

int tk_aes_encode(uint8_t* plain_text, int plain_length, std::string & encrypt_buffer, uint8_t* key, uint8_t*iv);

int tk_aes_decode(uint8_t*encrypt_buffer, int encrypt_length, std::string & plain_text_buffer, uint8_t* key, uint8_t*iv);

#endif // !__COMMON_H__