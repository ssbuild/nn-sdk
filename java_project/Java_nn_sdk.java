package nn_sdk;

public class Java_nn_sdk{


	
	public native static int  sdk_init_cc();
	public native static int  sdk_uninit_cc();
	public native static long sdk_new_cc(String json);
	public native static int  sdk_delete_cc(long handle);

	static {
		System.load("D:\\Python\\Python38\\Lib\\site-packages\\nn_sdk\\tf_csdk.pyd");  //以绝对路径加载so文件    
	}

	public static void main(String[] args){  
		System.out.println("java main...........");
	   sdk_init_cc();
	   sdk_uninit_cc();
	}

}