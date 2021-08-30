#ifndef __CC_SDK_H__
#define __CC_SDK_H__

#include <stdio.h>


#ifdef _WIN32
#ifdef _CC_SDK_EXPORT
#define CC_SDK_EXPORT _declspec( dllexport )
#else
#define CC_SDK_EXPORT _declspec(dllimport)
#endif
#else
#define CC_SDK_EXPORT
#endif // _WIN32



#ifdef __cplusplus
extern "C" {
#endif

	typedef long long SDK_HANDLE_CC;

	CC_SDK_EXPORT int sdk_init_cc();

	CC_SDK_EXPORT int sdk_uninit_cc();

	CC_SDK_EXPORT SDK_HANDLE_CC sdk_new_cc(const char* json);

	CC_SDK_EXPORT int sdk_delete_cc(SDK_HANDLE_CC handle);

	CC_SDK_EXPORT int sdk_process_ex_cc(SDK_HANDLE_CC handle, void** final_result, int net_stage, void**input_buffer_list);

#ifdef __cplusplus
}
#endif


#endif
