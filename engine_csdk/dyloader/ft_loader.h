#ifndef __FT_LOADER_H__
#define __FT_LOADER_H__
#pragma once

    #include "dyloader.h"

    class C_dylink_ft : public C_dylink_module {
        public:
            static C_dylink_ft* Instance() {
                static C_dylink_ft inst;
                return &inst;
            }

        public:
            explicit C_dylink_ft(){}
            bool load(const char* szSoPath,int engine_version);
    };

    struct S_ft_param_inout {
        const char* model_dir;
        int model_size{ -1 };
        int is_model_dir_file{ 1 };
        int dump_label{1};
        int debug{ 0 };
        int vector_dim;
        int version;
        void* labels{NULL}; //std::vecor<std::string>**
    };


    typedef int (*f_ft_new)(S_ft_param_inout& inout, void* &handle);
	typedef int (*f_ft_destroy)(void* handle);
	typedef int (*f_ft_process)(void* handle, char** text_utf8, int n, int batch, void** output_buf_only_read, int* out_buf_size);
    typedef int  (*f_ft_process_label)(void* handle, char** text_utf8, int n, int batch, void** output_buf_only_read, int k, int threshold);

	extern f_ft_new ft_new;
    extern f_ft_destroy ft_destroy;
    extern f_ft_process ft_process;
    extern f_ft_process_label ft_process_label;

#endif