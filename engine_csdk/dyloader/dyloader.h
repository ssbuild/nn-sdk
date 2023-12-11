#ifndef __DYLOADER_H__
#define __DYLOADER_H__

#include <stdint.h>
#include <time.h>
#include <vector>
#include <stdarg.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define __try  try
#define __except(x) catch(...)
//typedef long long   __int64;
#ifdef _WIN32

#else
#define _T
#include <dlfcn.h> 
#define sprintf_s snprintf
#endif // !_WIN32


class C_dylink_module {
public:
	bool valid{ 0 };
	C_dylink_module();
	virtual ~C_dylink_module();
protected:
	void* handle{ NULL };
};

#endif