/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class nn_sdk_Java_nn_sdk */

#ifndef _Included_nn_sdk_Java_nn_sdk
#define _Included_nn_sdk_Java_nn_sdk
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     nn_sdk_Java_nn_sdk
 * Method:    sdk_init_cc
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_nn_1sdk_Java_1nn_1sdk_sdk_1init_1cc
  (JNIEnv *, jclass);

/*
 * Class:     nn_sdk_Java_nn_sdk
 * Method:    sdk_uninit_cc
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_nn_1sdk_Java_1nn_1sdk_sdk_1uninit_1cc
  (JNIEnv *, jclass);

/*
 * Class:     nn_sdk_Java_nn_sdk
 * Method:    sdk_new_cc
 * Signature: (Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_nn_1sdk_Java_1nn_1sdk_sdk_1new_1cc
  (JNIEnv *, jclass, jstring);

/*
 * Class:     nn_sdk_Java_nn_sdk
 * Method:    sdk_delete_cc
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_nn_1sdk_Java_1nn_1sdk_sdk_1delete_1cc
  (JNIEnv *, jclass, jlong);

#ifdef __cplusplus
}
#endif
#endif
