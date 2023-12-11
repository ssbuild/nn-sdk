#ifndef __TF_ENTITIES_H__
#define __TF_ENTITIES_H__
#pragma once

#include "numpy/arrayobject.h"
//#include <regex>

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <map>
#include <algorithm>
#include <numeric>

class C_entities_resource {
public:
	C_entities_resource(const char* labels_file_, int imode, int debug) {
		label_file = labels_file_;
		m_debug = debug;
		m_imode = imode;
	}
	virtual ~C_entities_resource() {
		vec_org_label.clear();
		labels.clear();
		labels2id.clear();
		id2labels.clear();
	}

	int init() {
		if (!get_label_from_file()) {
			return -1;
		}
		if (m_imode == 0) {
			bool is_label_need_convert = false;
			for (auto & x : vec_org_label) {
				if (x.size() > 0) {
					if (x[0] == 'O' || x[0] == 'B' || x[0] == 'I' || x[0] == 'S' || x[0] == 'M') {
					}
					else {
						is_label_need_convert = true;
						break;
					}
				}
			
			}
			if (!is_label_need_convert) {
				labels = vec_org_label;
			}
			else {
				std::string str_array[] = {
					"B-","I-","S-"
				};
				labels.push_back(std::string("O"));
				for (int i = 0; i < sizeof(str_array) / sizeof(str_array[0]); ++i) {
					for (int j = 0; j < vec_org_label.size(); ++j) {
						labels.push_back(str_array[i] + vec_org_label[j]);
					}
				}
			}
		}
		else {
			labels = vec_org_label;
		}
		m_num_label = (int)labels.size();

		int i = 0;
		for (auto & it : labels) {
			labels2id.insert(std::make_pair(it, i));
			id2labels.insert(std::make_pair(i, it));
			i++;
			if (m_debug) {
				printf("labels: %s => %d \n", it.c_str(), i);
			}
		}
		return 0;
	}
protected:
	bool get_label_from_file();
private:
	std::vector<std::string>vec_org_label;
public:
	
	std::vector<std::string>labels;
	std::map<std::string, int> labels2id;
	std::map<int, std::string>id2labels;
	int m_debug;
	int m_imode;
	std::string label_file;
	int m_num_label;


	std::vector<std::string*>m_ner_ids;

};
C_entities_resource* tf_sdk_entities_new(const char* labels_file, int imode, int debug);
int tf_sdk_get_entities(C_entities_resource* resource, PyObject* preds, PyObject* mask, PyObject** result);
int tf_sdk_entities_delete(C_entities_resource* resource);


#endif // !__TF_ENTITIES_H__

