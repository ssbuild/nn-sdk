#include <stdio.h>  
#include <stdlib.h>
#include <cstring>
#include <dlfcn.h>  
typedef long long SDK_HANDLE_CC;

typedef int (*_sdk_init_cc)();
typedef int (*_sdk_uninit_cc)();
typedef SDK_HANDLE_CC (*_sdk_new_cc)(const char* json);
typedef int (*_sdk_delete_cc)(SDK_HANDLE_CC handle);
typedef int (*_sdk_process_cc)(SDK_HANDLE_CC handle, int net_stage,int batch_size, void** input_buffer_list,void** final_result);

_sdk_init_cc sdk_init_cc;
_sdk_uninit_cc sdk_uninit_cc;
_sdk_new_cc sdk_new_cc;
_sdk_delete_cc sdk_delete_cc;
_sdk_process_cc sdk_process_cc;


int test1() {
	if (0 != sdk_init_cc()) {
		return -1;
	}
	printf("main.........\n");
	//E:/algo_text/nn_csdk/nn_csdk/py_test_ckpt/model.ckpt
	    const char* json_data = "{\n\
    \"model_dir\": \"/root/model.ckpt\",\n\
    \"log_level\":8, \n\
    \"tf\":{ \n\
         \"ConfigProto\": {\n\
            \"log_device_placement\":0,\n\
            \"allow_soft_placement\":1,\n\
            \"gpu_options\":{\"allow_growth\": 1}\n\
        },\n\
        \"engine_version\": 1,\n\
        \"model_type\":1 \n\
    },\n\
    \"graph\": [\n\
        {\n\
            \"input\": [{\"node\":\"input_ids:0\", \"data_type\":\"float\", \"shape\":[1, 10]}],\n\
            \"output\" : [{\"node\":\"pred_ids:0\", \"data_type\":\"float\", \"shape\":[1, 10]}]\n\
        }\n\
    ]\n\
}";
	printf("%s\n", json_data);
	auto handle = sdk_new_cc(json_data);
	const int INPUT_NUM = 1;
	const int OUTPUT_NUM = 1;
	const int M = 1;
	const int N = 10;
	int *input[INPUT_NUM] = { 0 };
	float* result[OUTPUT_NUM] = { 0 };
	int element_input_size = sizeof(int);
	int element_output_size = sizeof(float);
	for (int i = 0; i < OUTPUT_NUM; ++i) {
		result[i] = (float*)malloc(M * N * element_output_size);
		memset(result[i], 0, M * N * element_output_size);
	}
	for(int i =0;i<INPUT_NUM;++i){
		input[i] = (int*)malloc(M * N * element_input_size);
		memset(input[i], 0, M * N * element_input_size);
		for (int j = 0; j < N; ++j) {
			input[i][j] = i;
		}
	}

	int code = sdk_process_cc(handle,  0, 1,(void**)input,(void**)result);
	if (code == 0) {
		printf("result\n");
		for (int i = 0; i < N; ++i) {
			printf("%f ", result[0][i]);
		}
		printf("\n");
	}
	for (int i = 0; i < INPUT_NUM; ++i) {
		free(input[i]);
	}
	for (int i = 0; i < OUTPUT_NUM; ++i) {
		free(result[i]);
	}
	sdk_delete_cc(handle);
	sdk_uninit_cc();
	return 0;
}



int main() {
	char *error;
	void *handle = dlopen("./engine_csdk.so", RTLD_LAZY | RTLD_GLOBAL);
	if (!handle) {
		fprintf(stderr, "%s ", dlerror());
		exit(1);
	}

	sdk_init_cc = (_sdk_init_cc)dlsym(handle, "sdk_init_cc");
	sdk_uninit_cc = (_sdk_uninit_cc)dlsym(handle, "sdk_uninit_cc");
	sdk_new_cc = (_sdk_new_cc)dlsym(handle, "sdk_new_cc");
	sdk_delete_cc = (_sdk_delete_cc)dlsym(handle, "sdk_delete_cc");
	sdk_process_cc = (_sdk_process_cc)dlsym(handle, "sdk_process_cc");

	
	if ((error = dlerror()) != NULL) {
		fprintf(stderr, "%s ", error);
		exit(1);
	}

	if (sdk_init_cc == NULL ||
		sdk_uninit_cc == NULL ||
		sdk_new_cc == NULL ||
		sdk_delete_cc == NULL ||
		sdk_process_cc == NULL)
	{
		fprintf(stderr, "function no load\n");
		exit(1);
	}
	
	test1();

	dlclose(handle);
	return 0;
}