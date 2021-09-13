package nn_sdk;
//包名必须是nn_sdk
public class nn_sdk {
    //输入输出内存节点，名字跟图配置一样，根据图对象修改此属性。 
	public float [] input_ids = null;//推理图的输入,
	//public float [] input_mask = null;//推理图的输入,
	public float[] pred_ids =   null;//推理的结果保存

	public nn_sdk() {
		//初始化配置图的输入输出内存
		input_ids = new float[1 * 20];
		pred_ids =  new float[1 * 20];
		for(int i =0;i<20;i++) {
			input_ids[i] = 1;
			pred_ids[i] = 0;
		}
	}
	
	//推理函数
	public native static int  sdk_init_cc();
	public native static int  sdk_uninit_cc();
	public native static long sdk_new_cc(String json);
	public native static int  sdk_delete_cc(long handle);
	public native static int sdk_process_cc(long handle, int net_state, nn_sdk data);
	
	static {
		//动态库的绝对路径windows是 engine_csdk.pyd , linux是 engine_csdk.so
		System.load("E:\\algo_text\\nn_csdk\\build_py36\\Release\\engine_csdk.pyd");
	}

	public static void main(String[] args){  
		System.out.println("java main...........");
		
	   nn_sdk instance = new nn_sdk();
	   sdk_init_cc();
	   //配置参考 python
	   String json =  "{" + "\"model_dir\": \"E:/algo_text/nn_csdk/nn_csdk/py_test_ckpt/model.ckpt\"," + "\n" +
	   "\"log_level\": 4," + "\n" +
	   "\"engine\": 0," + "\n" +
	   "\"device_id\": 0," + "\n" +
	   "\"tf\":{ " + "\n" +
           "\"ConfigProto\": {" + "\n" +
                     "\"log_device_placement\":false," + "\n" +
                      "\"allow_soft_placement\":true," + "\n" +
                     "\"gpu_options\":{\"allow_growth\": true}" + "\n" +
           "}," + "\n" +
			"\"engine_version\": 1," + "\n" +
			"\"model_type\":1, " + "\n" +
	    "}" + "\n" +
	   "\"graph\": [" + "\n" +
				    "{" + "\n" +
                        "\"input\": [{\"node\":\"input_ids:0\", \"data_type\":\"float\", \"shape\":[1, 20]}]," + "\n" +
                        "\"output\" : [{\"node\":\"pred_ids:0\", \"data_type\":\"float\", \"shape\":[1, 20]}]" + "\n" +
	                "}" + "\n" +
		        "]" + "\n" +
	    "}";
				

	  System.out.println(json);
	   
	  long handle = sdk_new_cc(json);
	  System.out.println(handle);
	   
	  int code = sdk_process_cc(handle,0,instance);
	  System.out.printf("sdk_process_cc %d \n" ,code);
	  if(code == 0) {
		  for(int i = 0;i<20 ; i++) {
			  System.out.printf("%f ",instance.pred_ids[i]);
		  }
		  System.out.println();
	  }
	  sdk_delete_cc(handle);
	   sdk_uninit_cc();
	   System.out.println("end");
	}
}
