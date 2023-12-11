#include "../common.h"
#include "dyloader.h"
#ifdef _WIN32
#include<Windows.h>
#endif // _WIN32




C_dylink_module::C_dylink_module() { valid = false; }

C_dylink_module::~C_dylink_module() {
#ifdef _WIN32
	if (handle) {
		FreeLibrary((HMODULE)handle);
		handle = NULL;
	}
#else
	if (handle) {
		dlclose(handle);
	}
#endif // __WIN32
}





