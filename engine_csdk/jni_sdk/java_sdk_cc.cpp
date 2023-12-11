#include "nn_sdk_nn_sdk.h"
#include "../common.h"
#include "../nn_sdk.h"


JNIEXPORT jint JNICALL Java_nn_1sdk_nn_1sdk_sdk_1init_1cc
(JNIEnv *, jclass) {

	return sdk_init_cc();
}


JNIEXPORT jint JNICALL Java_nn_1sdk_nn_1sdk_sdk_1uninit_1cc
(JNIEnv *, jclass) {
	return sdk_uninit_cc();
}


JNIEXPORT jlong JNICALL Java_nn_1sdk_nn_1sdk_sdk_1new_1cc
(JNIEnv * env, jclass java_class, jstring j_json) {

	jboolean iscopy = 1;
	const char* json = env->GetStringUTFChars(j_json, &iscopy);
	auto handle = sdk_new_cc(json);
	env->ReleaseStringUTFChars(j_json, json);
	return handle;
}


JNIEXPORT jint JNICALL Java_nn_1sdk_nn_1sdk_sdk_1delete_1cc
(JNIEnv *, jclass, jlong handle) {
	return sdk_delete_cc(handle);
}

int get_java_arr_pts(JNIEnv* env, jclass java_class, jobject jobj, const char* name, int data_type, void*& arr_object, void*& arr_elements) {
	jfieldID fid_arrays = NULL;
	if (data_type == NPY_INT ||
		data_type == NPY_LONG ||
		data_type == NPY_UINT ||
		data_type == NPY_ULONG) {
		fid_arrays = env->GetFieldID(java_class, name, "[I");
		jarray arr = (jarray)env->GetObjectField(jobj, fid_arrays);
		arr_object = arr;
		arr_elements = env->GetPrimitiveArrayCritical(arr, NULL);
	}
	else if (data_type == NPY_ULONGLONG || data_type == NPY_LONGLONG) {
		fid_arrays = env->GetFieldID(java_class, name, "[J");
		jarray arr = (jarray)env->GetObjectField(jobj, fid_arrays);
		arr_object = arr;
		arr_elements = env->GetPrimitiveArrayCritical(arr, NULL);
	}
	else if (data_type == NPY_FLOAT) {
		fid_arrays = env->GetFieldID(java_class, name, "[F");
		jarray arr = (jarray)env->GetObjectField(jobj, fid_arrays);
		arr_object = arr;
		arr_elements = env->GetPrimitiveArrayCritical(arr, NULL);
	}
	else if (data_type == NPY_DOUBLE) {
		fid_arrays = env->GetFieldID(java_class, name, "[D");
		jarray arr = (jarray)env->GetObjectField(jobj, fid_arrays);
		arr_object = arr;
		arr_elements = env->GetPrimitiveArrayCritical(arr, NULL);
	}
	else {
		return -1;
	}
	if (arr_elements == NULL) {
		return -1;
	}
	return 0;
}
void release_java_arr_pts(JNIEnv* env, void* arr_object, void* arr_elements, int data_type) {
	if (arr_object && arr_elements) {
		env->ReleasePrimitiveArrayCritical((jarray)arr_object, (void*)arr_elements, 0);
	}
}

int get_java_arr(JNIEnv * env, jclass java_class, jobject jobj,const char* name , int data_type,void* & arr_object,void* & arr_elements) {
	jfieldID fid_arrays = NULL;
	if (data_type == NPY_INT ||
		data_type == NPY_LONG ||
		data_type == NPY_UINT ||
		data_type == NPY_ULONG) {
		fid_arrays = env->GetFieldID(java_class, name, "[I");
		jintArray arr = (jintArray)env->GetObjectField(jobj, fid_arrays);
		arr_object = arr;
		arr_elements = env->GetIntArrayElements(arr, NULL);
	}
	else if (data_type == NPY_ULONGLONG || data_type == NPY_LONGLONG) {
		fid_arrays = env->GetFieldID(java_class, name, "[J");
		jlongArray arr = (jlongArray)env->GetObjectField(jobj, fid_arrays);
		arr_object = arr;
		arr_elements = env->GetLongArrayElements(arr, NULL);
	}
	else if (data_type == NPY_FLOAT) {
		fid_arrays = env->GetFieldID(java_class, name, "[F");
		jfloatArray arr = (jfloatArray)env->GetObjectField(jobj, fid_arrays);
		arr_object = arr;
		arr_elements = env->GetFloatArrayElements(arr, NULL);
	}
	else if (data_type == NPY_DOUBLE) {
		fid_arrays = env->GetFieldID(java_class, name, "[D");
		jdoubleArray arr = (jdoubleArray)env->GetObjectField(jobj, fid_arrays);
		arr_object = arr;
		arr_elements = env->GetDoubleArrayElements(arr, NULL);
	}
	else {
		return -1;
	}
	return 0;
}

void release_java_arr(JNIEnv * env,void* arr_object,void*arr_elements,int data_type) {
	if (arr_object && arr_elements) {
		if (data_type == NPY_INT ||
			data_type == NPY_LONG ||
			data_type == NPY_UINT ||
			data_type == NPY_ULONG) {
			env->ReleaseIntArrayElements((jintArray)arr_object, (jint*)arr_elements, 0);
		}
		else if (data_type == NPY_ULONGLONG || data_type == NPY_LONGLONG) {
			env->ReleaseLongArrayElements((jlongArray)arr_object, (jlong*)arr_elements, 0);
		}
		else if (data_type == NPY_FLOAT) {
			env->ReleaseFloatArrayElements((jfloatArray)arr_object, (jfloat*)arr_elements, 0);
		}
		else if (data_type == NPY_DOUBLE) {
			env->ReleaseDoubleArrayElements((jdoubleArray)arr_object, (jdouble*)arr_elements, 0);
		}
	}
}

JNIEXPORT jint JNICALL Java_nn_1sdk_nn_1sdk_sdk_1process_1cc__old
(JNIEnv * env, jclass, jlong handle, jint net_stage,jint batch_size, jobject jobj){

	jint code = -1;
	jclass java_class = env->GetObjectClass(jobj);
	if (!java_class) {
		log_err("%s GetObjectClass failed",__FUNCTION__);
		return code;
	}
	C_engine_base* resource = (C_engine_base*)handle;
	if (net_stage > resource->m_net_graph.size()) {
		log_err("%s bad net_stage:%d\n",__FUNCTION__, net_stage);
		return code;
	}
	int data_type = 0;
	bool has_error = false;
	size_t pos = -1;
	std::string name;
	auto & curr_net = resource->m_net_graph[net_stage];
	void **input_data_list = NULL;
	void **output_data_list = NULL;
	output_data_list = (void **)malloc(curr_net.output_.size() * 2 * sizeof(void*));
	input_data_list = (void **)malloc(curr_net.input_.size() * 2 * sizeof(void*));
	if (!output_data_list || !input_data_list) {
		if (output_data_list) free(output_data_list);
		if (input_data_list) free(input_data_list);
		log_err("%s malloc failed", __FUNCTION__);
		return code;
	}
	for (int i = 0; i < curr_net.output_.size(); ++i) {
		output_data_list[i] = 0;
	}
	for (int i = 0; i < curr_net.input_.size(); ++i) {
		input_data_list[i] = 0;
	}

	for (int i = 0; i < curr_net.output_.size(); ++i) {
		auto& net_output = curr_net.output_[i];
		//java 实际指针
		auto & arr_elements = output_data_list[i];
		//java 实际指针obj
		auto & arr_object = output_data_list[i + curr_net.output_.size()];
		
		pos = net_output.name.find(':');
		if (pos >= 0) {
			name = net_output.name.substr(0, pos);
		}
		else {
			name = net_output.name;
		}
		data_type = net_output.data_type;
		if (0 != get_java_arr(env, java_class, jobj, name.c_str(), data_type, arr_object, arr_elements)) {
			log_err("%s output %s bad data_type", __FUNCTION__, name.c_str());
			has_error = true;
			break;
		}
	}
	if (has_error) {
		goto End;
	}

	for (int i = 0; i < curr_net.input_.size(); ++i) {
		auto & arr_elements = input_data_list[i];
		auto & arr_object = input_data_list[i + curr_net.input_.size()];
		auto & net_input = curr_net.input_[i];
		pos = net_input.name.find(':');
		if (pos >= 0) {
			name = net_input.name.substr(0, pos);
		}
		else {
			name = net_input.name;
		}
		data_type = net_input.data_type;
		if (0 != get_java_arr(env, java_class, jobj, name.c_str(), data_type, arr_object, arr_elements)) {
			log_err("%s output %s bad data_type", __FUNCTION__, name.c_str());
			has_error = true;
			break;
		}
	}

	if (has_error) {
		goto End;
	}
	code = sdk_process_cc(handle, net_stage, batch_size, input_data_list, output_data_list);
End:
	//清理output
	if (output_data_list) {
		for (int i = 0; i < curr_net.output_.size(); ++i) {
			auto & net_output = curr_net.output_[i];
			auto & arr_elements = output_data_list[i];
			auto & arr_object = output_data_list[i + curr_net.output_.size()];
			release_java_arr(env, arr_object, arr_elements, net_output.data_type);
		}
		free(output_data_list);
	}
	//清理input
	if (input_data_list) {
		for (int i = 0; i < curr_net.input_.size(); ++i) {
			auto & net = curr_net.input_[i];
			auto & arr_elements = input_data_list[i];
			auto & arr_object = input_data_list[i + curr_net.input_.size()];
			release_java_arr(env, arr_object, arr_elements, net.data_type);
		}
		free(input_data_list);
	}
	return code;
}







//新方法


JNIEXPORT jint JNICALL Java_nn_1sdk_nn_1sdk_sdk_1process_1cc
(JNIEnv* env, jclass, jlong handle, jint net_stage, jint batch_size, jobject jobj) {

	jint code = -1;
	jclass java_class = env->GetObjectClass(jobj);
	if (!java_class) {
		log_err("%s GetObjectClass failed", __FUNCTION__);
		return code;
	}
	C_engine_base* resource = (C_engine_base*)handle;
	if (net_stage > resource->m_net_graph.size()) {
		log_err("%s bad net_stage:%d\n", __FUNCTION__, net_stage);
		return code;
	}
	int data_type = 0;
	bool has_error = false;
	size_t pos = -1;
	std::string name;
	auto& curr_net = resource->m_net_graph[net_stage];
	void** input_data_list = NULL;
	void** output_data_list = NULL;
	output_data_list = (void**)malloc(curr_net.output_.size() * 2 * sizeof(void*));
	input_data_list = (void**)malloc(curr_net.input_.size() * 2 * sizeof(void*));
	if (!output_data_list || !input_data_list) {
		if (output_data_list) free(output_data_list);
		if (input_data_list) free(input_data_list);
		log_err("%s malloc failed\n", __FUNCTION__);
		return code;
	}
	for (int i = 0; i < curr_net.output_.size(); ++i) {
		output_data_list[i] = 0;
	}
	for (int i = 0; i < curr_net.input_.size(); ++i) {
		input_data_list[i] = 0;
	}

	for (int i = 0; i < curr_net.output_.size(); ++i) {
		auto& net_output = curr_net.output_[i];
		//java 实际指针
		auto& arr_elements = output_data_list[i];
		//java 实际指针obj
		auto& arr_object = output_data_list[i + curr_net.output_.size()];

		pos = net_output.name.find(':');
		if (pos >= 0) {
			name = net_output.name.substr(0, pos);
		}
		else {
			name = net_output.name;
		}
		data_type = net_output.data_type;
		if (0 != get_java_arr_pts(env, java_class, jobj, name.c_str(), data_type, arr_object, arr_elements)) {
			log_err("%s output %s bad data_type\n", __FUNCTION__, name.c_str());
			has_error = true;
			break;
		}
	}
	if (has_error) {
		goto End;
	}

	for (int i = 0; i < curr_net.input_.size(); ++i) {
		auto& net_input = curr_net.input_[i];

		auto& arr_elements = input_data_list[i];
		auto& arr_object = input_data_list[i + curr_net.input_.size()];
		pos = net_input.name.find(':');
		if (pos >= 0) {
			name = net_input.name.substr(0, pos);
		}
		else {
			name = net_input.name;
		}
		data_type = net_input.data_type;
		if (0 != get_java_arr_pts(env, java_class, jobj, name.c_str(), data_type, arr_object, arr_elements)) {
			log_err("%s input %s bad data_type\n", __FUNCTION__, name.c_str());
			has_error = true;
			break;
		}
	}

	if (has_error) {
		goto End;
	}
	code = sdk_process_cc(handle, net_stage, batch_size, input_data_list, output_data_list);
End:
	//清理output
	if (output_data_list) {
		for (int i = 0; i < curr_net.output_.size(); ++i) {
			auto& net_output = curr_net.output_[i];
			auto& arr_elements = output_data_list[i];
			auto& arr_object = output_data_list[i + curr_net.output_.size()];
			release_java_arr_pts(env, arr_object, arr_elements, net_output.data_type);
		}
		free(output_data_list);
	}
	//清理input
	if (input_data_list) {
		for (int i = 0; i < curr_net.input_.size(); ++i) {
			auto& net = curr_net.input_[i];
			auto& arr_elements = input_data_list[i];
			auto& arr_object = input_data_list[i + curr_net.input_.size()];
			release_java_arr_pts(env, arr_object, arr_elements, net.data_type);
		}
		free(input_data_list);
	}
	return code;
}