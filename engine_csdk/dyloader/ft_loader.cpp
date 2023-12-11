#include "../common.h"
#include "ft_loader.h"
#ifdef _WIN32
    #include<Windows.h>
#endif

f_ft_new ft_new = NULL;
f_ft_destroy ft_destroy = NULL;
f_ft_process ft_process = NULL;
f_ft_process_label ft_process_label = NULL;



bool C_dylink_ft::load(const char* szSoPath, int engine_version) {
	if (valid) {
		return valid;
	}
	char* error;
	char sz_path[255] = { 0 };
#ifdef _WIN32
	sprintf_s(sz_path, sizeof(sz_path), "%s/fasttext_inf.dll", szSoPath);
#else
	sprintf_s(sz_path, sizeof(sz_path), "%s/fasttext_inf.so", szSoPath);
#endif
	log_debug("%s\n", sz_path);
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
	*(void**)&ft_new = GetProcAddress((HMODULE)handle, "ft_new");
	*(void**)&ft_destroy = GetProcAddress((HMODULE)handle, "ft_destroy");
	*(void**)&ft_process = GetProcAddress((HMODULE)handle, "ft_process");
	*(void**)&ft_process_label = GetProcAddress((HMODULE)handle, "ft_process_label");
	

#else

	dlerror();

	handle = dlopen(sz_path, RTLD_NOW);
	if (!handle) {
		valid = false;
		log_err("open %s failed,err=%s\n", sz_path, dlerror());
		return valid;
	}


	*(void**)&ft_new = dlsym(handle, "ft_new");
	*(void**)&ft_destroy = dlsym(handle, "ft_destroy");
	*(void**)&ft_process = dlsym(handle, "ft_process");
	*(void**)&ft_process_label = dlsym(handle, "ft_process_label");
	

	if ((error = dlerror()) != NULL) {
		valid = false;
		log_err(_T("path %s , open %s failed\n"), sz_path, error);
		return valid;
	}
#endif // __WIN32
	valid = true;
	return valid;
}