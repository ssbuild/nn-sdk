#ifndef __TRT_LOADER_H__
#define __TRT_LOADER_H__
#include "dyloader.h"

class C_dylink_trt : public C_dylink_module {
public:
	static C_dylink_trt* Instance() {
		static C_dylink_trt inst;
		return &inst;
	}

public:
	explicit C_dylink_trt(){}
	bool load(const char* szSoPath,int engine_major,int engine_minor);
};


typedef void* CC_SDK_TRT_HANDLE;
typedef int (*f_cc_sdk_trt_delete)(const CC_SDK_TRT_HANDLE instance);
typedef int (*f_cc_sdk_trt_new)(const char* model_buffer, int buffer_size,
	int device_id,
	const bool enableGraph,
	std::vector<std::vector<std::vector<int>>>& input_shape_,
	std::vector<std::vector<int>>& input_item_size_,
	std::vector<std::vector<std::vector<int>>>& output_shape_,
	std::vector<std::vector<int>>& output_item_size_,
	CC_SDK_TRT_HANDLE& instance);


typedef int (*f_cc_sdk_trt_process)(const CC_SDK_TRT_HANDLE instance, int net_stage, const void* const* input_buffer_list,
	const int input_num, const int batch_size, void** output_buf_only_read, int* output_buf_size);


extern f_cc_sdk_trt_new cc_sdk_trt_new;
extern f_cc_sdk_trt_delete cc_sdk_trt_delete;
extern f_cc_sdk_trt_process cc_sdk_trt_process;

#endif