#include "tf_entities.h"


typedef struct S_entities_item {
	int s;
	int e;
	std::string label;
}*PS_entities_item;


static inline void ltrim(std::string &s) {
	s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
		return !isspace(ch);
	}));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
	s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
		return !isspace(ch);
	}).base(), s.end());
}

// trim from both ends (in place)
static inline void trim(std::string &s) {
	ltrim(s);
	rtrim(s);
}



bool C_entities_resource::get_label_from_file() {
	std::ifstream in(label_file);
	if (in) {
		int i = 0;
		std::string line;
		//std::regex pattern(" |\n|\r|\t");
		while (std::getline(in, line)) {
			trim(line);
			if (line == "" || line.empty()) continue;
			if (m_debug) {
				std::cout << line << std::endl;
			}
			//line = std::regex_replace(line, pattern, "");
			vec_org_label.push_back(line);
			i++;
		}
		in.close();
		return true;
	}
	std::cout << "no file: " << label_file << std::endl;
	return false;
}

void get_entities(std::vector<std::string*>& ner_ids,const int length,std::vector<S_entities_item> & ner_out) {
	//auto length = ner_ids.size();
	S_entities_item chunk = {-1, -1};
	chunk.s = -1;
	chunk.e = -1;

	for (int i = 0; i < length; ++i) {
		auto tag = ner_ids[i];
		auto & first = tag->at(0);

		if (first == 'S') {
			if (chunk.e != -1) {
				ner_out.push_back(chunk);
			}
			chunk.s = i;
			chunk.e = i;
			chunk.label = tag->substr(2);
			ner_out.push_back(chunk);

			chunk.s = -1;
			chunk.e = -1;
			chunk.label.clear();

		}else if (first == 'B') {
			if (chunk.e != -1) {
				ner_out.push_back(chunk);
			}

			chunk.s = -1;
			chunk.e = -1;
			chunk.label.clear();
			chunk.s = i;
			chunk.label = tag->substr(2);
		}
		else if (first == 'I') {
			if (chunk.s != -1) {
				if (tag->substr(2) == chunk.label) {
					chunk.e = i;
				}
				if (i == length - 1) {
					ner_out.push_back(chunk);
				}
			}
		}
		else {
			if (chunk.e != -1) {
				ner_out.push_back(chunk);
			}
			chunk.s = -1;
			chunk.e = -1;
			chunk.label.clear();
		}
	}
}


C_entities_resource* tf_sdk_entities_new(const char* labels_file, int imode, int debug) {
	C_entities_resource* resource = new C_entities_resource(labels_file, imode, debug);
	if (!resource || resource->init() != 0) {
		if (resource)
			delete resource;
		return NULL;
	}
	return resource;
}

int tf_sdk_get_entities(C_entities_resource* resource, PyObject* preds,PyObject* mask, PyObject** result) {

	PyArrayObject *ListItem = (PyArrayObject*)preds;
	int ndim = PyArray_NDIM(ListItem);
	npy_intp* dims = PyArray_DIMS(ListItem);

	npy_intp* strides = NULL;
	char* data = NULL;

	PyObject* ret_list = NULL;

	if (resource->m_imode == 0) {
		if (ndim != 2 && ndim != 3) {
			return -1;
		}
		PyArrayObject* newList = NULL;
		if (ndim == 3) {
			newList = (PyArrayObject*)PyArray_ArgMax(ListItem, 2, NULL);
			if (!newList) {
				return -1;
			}
			ListItem = newList;
		}

		int rows = (int)dims[0], columns = 0;
		ret_list = PyList_New(rows);
		*result = ret_list;

		strides = PyArray_STRIDES(ListItem);
		data = (char*)PyArray_BYTES(ListItem);
		//dims = PyArray_DIMS(ListItem);

		columns = (int)dims[1];

		resource->m_ner_ids.clear();
		resource->m_ner_ids.resize(rows);

		auto & pos = resource->m_ner_ids;
		pos.resize(columns);
		std::vector<S_entities_item> ner_out;

		long long mask_value = 0;

		for (int i = 0; i < rows; i++) {
			int pos_length = 0;

			auto oMask_list = PyList_GetItem(mask, i);
			if (!oMask_list) {
				PyErr_Print();
				return -1;
			}
			for (int j = 0; j < columns; j++) {
				auto oItem = PyList_GetItem(oMask_list, j + 1);
				if (!oItem) {
					PyErr_Print();
					return -1;
				}
				mask_value = PyLong_AsLongLong(oItem);
				if (mask_value == 0) {
					break;
				}
				auto & value = *(long long *)(data + i * strides[0] + j * strides[1]);
				pos[j] = &resource->id2labels[(int)value];

				++pos_length;
			}

			ner_out.clear();
			get_entities(pos, pos_length, ner_out);

			auto onelist = PyList_New(ner_out.size());
			Py_ssize_t j = -1;
			for (auto & x : ner_out) {
				++j;
				PyList_SetItem(onelist, j, Py_BuildValue("[s#,i,i]", x.label.c_str(), x.label.length(), x.s, x.e));
			}
			PyList_SetItem(ret_list, i, onelist);
		}
		if (newList) {
			Py_DECREF(newList);
		}
	}
	else if (resource->m_imode == 1) {

		if (ndim != 1 && ndim != 2) {
			return -1;
		}
		strides = PyArray_STRIDES(ListItem);
		data = (char*)PyArray_BYTES(ListItem);
		int rows = (int)dims[0];
		ret_list = PyList_New(rows);
		if (ret_list == NULL) {
			return -1;
		}
		*result = ret_list;

		if (ndim == 2) {
			int columns = (int)dims[1];
			float value = 0;
			float max_f = -INFINITY;
			int max_index = 0;
			for (int i = 0; i < rows; i++) {
				max_f = -INFINITY;
				max_index = 0;
				for (int j = 0; j < columns; j++) {
					value = *(float *)(data + i * strides[0] + j * strides[1]);
					if (value > max_f) {
						max_f = value;
						max_index = j;
					}
				}
				PyList_SetItem(ret_list, i, Py_BuildValue("s", resource->id2labels[max_index].c_str()));
			}
		}
		else {
			for (int i = 0; i < rows; i++) {
				auto & value = *(long long *)(data + i * strides[0]);
				PyList_SetItem(ret_list, i, Py_BuildValue("s", resource->id2labels[value].c_str()));
			}
		
		}
		//PyArrayObject* newList = NULL;
		//if (ndim == 2) {
		//	newList = (PyArrayObject*)PyArray_ArgMax(ListItem, 1, NULL);
		//	if (!newList) {
		//		Py_DECREF(ret_list);
		//		return -1;
		//	}
		//	ListItem = newList;
		//}

		//
		//strides = PyArray_STRIDES(ListItem);
		//data = (char*)PyArray_BYTES(ListItem);
		////dims = PyArray_DIMS(ListItem);

		//for (int i = 0; i < rows; i++) {
		//	auto & value = *(long long *)(data + i * strides[0]);
		//	PyList_SetItem(ret_list, i, Py_BuildValue("s", resource->id2labels[value].c_str()));
		//}
		//if (newList) {
		//	Py_DECREF(newList);
		//}
	}
	return 0;
}

int tf_sdk_entities_delete(C_entities_resource* resource) {
	if (resource) {
		delete resource;
		return 0;
	}
	return -1;
}