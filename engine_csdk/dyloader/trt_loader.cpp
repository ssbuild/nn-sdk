#include "../common.h"
#include "trt_loader.h"
#ifdef _WIN32
    #include<Windows.h>
#endif

f_cc_sdk_trt_new cc_sdk_trt_new = NULL;
f_cc_sdk_trt_delete cc_sdk_trt_delete = NULL;
f_cc_sdk_trt_process cc_sdk_trt_process = NULL;


bool C_dylink_trt::load(const char* szSoPath, int engine_major,int engine_minor) {
	if (valid) {
		return valid;
	}
	char* error;
	char sz_path[255] = { 0 };

	sprintf_s(sz_path, sizeof(sz_path), "%s/engine_trt%d.%d.so", szSoPath, engine_major, engine_minor);

	if (access(sz_path, 0) != 0) {
		log_err("trt module file %s\n", sz_path);
		return false;
	}



	
	log_debug("dir %s ,so file %s\n", szSoPath, sz_path);
#ifdef _WIN32
	handle = LoadLibrary(sz_path);
	if (handle == NULL){
		LPVOID lpMsgBuf = NULL;
		auto dw = GetLastError();
		FormatMessageA(
			FORMAT_MESSAGE_ALLOCATE_BUFFER |
			FORMAT_MESSAGE_FROM_SYSTEM |
			FORMAT_MESSAGE_IGNORE_INSERTS,
			NULL,
			dw,
			MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
			(LPSTR)&lpMsgBuf,
			0, NULL);
		if (lpMsgBuf) {
			log_err("open %s failed,err=%s\n", sz_path, (LPCSTR)lpMsgBuf);
		}
		else {
			log_err("open %s failed\n", sz_path);
		}
		LocalFree(lpMsgBuf);
		return valid;
	}
	*(void**)&cc_sdk_trt_new = GetProcAddress((HMODULE)handle, "cc_sdk_trt_new");
	*(void**)&cc_sdk_trt_delete = GetProcAddress((HMODULE)handle, "cc_sdk_trt_delete");
	*(void**)&cc_sdk_trt_process = GetProcAddress((HMODULE)handle, "cc_sdk_trt_process");

#else

	dlerror();

	handle = dlopen(sz_path, RTLD_NOW);
	if (!handle) {
		valid = false;
		log_err("open %s failed,err=%s\n", sz_path, dlerror());
		return valid;
	}

	*(void**)&cc_sdk_trt_new = dlsym(handle, "cc_sdk_trt_new");
	*(void**)&cc_sdk_trt_delete = dlsym(handle, "cc_sdk_trt_delete");
	*(void**)&cc_sdk_trt_process = dlsym(handle, "cc_sdk_trt_process");

	if ((error = dlerror()) != NULL) {
		valid = false;
		log_err(_T("path %s , open %s failed\n"), sz_path, error);
		return valid;
	}
#endif // __WIN32
	valid = true;
	return valid;
}