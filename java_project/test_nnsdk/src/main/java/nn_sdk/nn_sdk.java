package nn_sdk;


//���뻺���� �Զ��� ���Զ����
//class nn_buffer_batch{
//	  //���� ����ڴ�ڵ㣬���ָ�ͼ����һ��������ͼ�����޸ġ�
//	public float [] input_ids = null;//����ͼ������,
//	public float[] pred_ids =   null;//����Ľ������
//	
//	public int batch_size = 1;
//	public nn_buffer_batch(int batch_size_){
//		this.input_ids = new float[batch_size_ * 10];
//		this.pred_ids =  new float[batch_size_ * 10];
//		this.batch_size = batch_size_;
//		for(int i =0;i<1 * 10;i++) {
//			this.input_ids[i] = 1;
//			this.pred_ids[i] = 0;
//		}
//	}
//}

class nn_buffer_batch{
	  //���� ����ڴ�ڵ㣬���ָ�ͼ����һ��������ͼ�����޸ġ�
	public float [] x1 = null;//����ͼ������,
	public float [] x2 = null;//����ͼ������,
	public float[] pred_ids =   null;//����Ľ������
	
	public int batch_size = 1;
	public nn_buffer_batch(int batch_size_){
		this.x1 = new float[batch_size_ * 10];
		this.x2 = new float[batch_size_ * 10];
		this.pred_ids =  new float[batch_size_ * 10];
		this.batch_size = batch_size_;
		for(int i =0;i<1 * 10;i++) {
			this.x1[i] = 1;
			this.x2[i] = 2;
			this.pred_ids[i] = 0;
		}
	}
}


//����������nn_sdk
public class nn_sdk {
	//������
	public native static int  sdk_init_cc();
	public native static int  sdk_uninit_cc();
	public native static long sdk_new_cc(String json);
	public native static int  sdk_delete_cc(long handle);
	//���Զ���nn_buffer_batch ��
	public native static int sdk_process_cc(long handle, int net_state,int batch_size, nn_buffer_batch buffer);
	
	static {
		//��̬��ľ���·��windows��engine_csdk.pyd , linux�� engine_csdk.so
		System.load("D:\\Python\\Python38\\Lib\\site-packages\\nn_sdk\\engine_csdk.pyd");  
	}

	public static void main(String[] args){  
		System.out.println("java main...........");
		
		
	   nn_sdk instance = new nn_sdk();
	   
	   nn_buffer_batch buf = new nn_buffer_batch(1);
	   sdk_init_cc();
	   
	   String json = "{\r\n"
	   + "    \"model_dir\": r'E:\\pypi_project\\nn-sdk\\train\\model.ckpt',\r\n"
	   + "    \"aes\":{\r\n"
	   + "        \"use\":False,\r\n"
	   + "        \"key\":bytes([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]),\r\n"
	   + "        \"iv\":bytes([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]),\r\n"
	   + "    },\r\n"
	   + "    \"log_level\": 8,# fatal 1 , error 2 , info 4 , debug 8\r\n"
	   + "    'engine':0, # 0 tensorflow,  1 onnx , 2  tensorrt , 3 fasttext\r\n"
	   + "    'tf':{\r\n"
	   + "        #tensorflow2 ConfigProto��Ч\r\n"
	   + "        \"ConfigProto\": {\r\n"
	   + "            \"log_device_placement\": 0,\r\n"
	   + "            \"allow_soft_placement\": 1,\r\n"
	   + "            \"gpu_options\": {\r\n"
	   + "                \"allow_growth\": 1\r\n"
	   + "            },\r\n"
	   + "            \"graph_options\":{\r\n"
	   + "                \"optimizer_options\":{\r\n"
	   + "                    \"global_jit_level\": 0\r\n"
	   + "                }\r\n"
	   + "            },\r\n"
	   + "        },\r\n"
	   + "        \"engine_version\": 1, # tensorflow�汾\r\n"
	   + "        \"model_type\": 1,# 0 pb , 1 ckpt\r\n"
	   + "        \"saved_model\":{ # ��model_typeΪpbģ����Ч, ��ͨpb use=False �� �����saved_model����ģ�� , ��������use��������tags\r\n"
	   + "            'use': False, # �Ƿ�����saved_model\r\n"
	   + "            'tags': ['serve'],\r\n"
	   + "            'signature_key': 'serving_default',\r\n"
	   + "        },\r\n"
	   + "        \"fastertransformer\":{\r\n"
	   + "            \"use\": False,\r\n"
	   + "            \"cuda_version\":\"11.3\", #��ǰ���� tf2pb,֧��10.2 11.3 ,\r\n"
	   + "        }\r\n"
	   + "    },\r\n"
	   + "    \"graph\": [\r\n"
	   + "        {\r\n"
	   + "            # ����Bertģ�� shape [max_batch_size,max_seq_lenth],\r\n"
	   + "            # ����max_batch_size ����c++ java���������������,���벻�ó���max_batch_size������pythonû�����ã�ȡ�����ϲ��û���ʵ����\r\n"
	   + "            # python����max_batch_size ���ϲ��û�������\r\n"
	   + "            # ����fasttext node ��Ӧname��������д����������\r\n"
	   + "            \"input\": [\r\n"
	   + "                {\"node\":\"x1:0\", \"data_type\":\"float\", \"shape\":[1, 10]},\r\n"
	   + "                {\"node\":\"x2:0\", \"data_type\":\"float\", \"shape\":[1, 10]},\r\n"
	   + "            ],\r\n"
	   + "            \"output\": [\r\n"
	   + "                {\"node\":\"pred_ids:0\", \"data_type\":\"float\", \"shape\":[1, 10]},\r\n"
	   + "            ],\r\n"
	   + "        }\r\n"
	   + "    ]}";
	   
				

	  //System.out.println(json);
	   
	  long handle = sdk_new_cc(json);
	  System.out.printf("handle: %d\n",handle);
	   
	  int code = sdk_process_cc(handle,0,buf.batch_size,buf);
	  System.out.printf("sdk_process_cc %d \n" ,code);
	  if(code == 0) {
		  for(int i = 0;i<10 ; i++) {
			  System.out.printf("%f ",buf.pred_ids[i]);
		  }
		  System.out.println();
	  }
	  sdk_delete_cc(handle);
	   sdk_uninit_cc();
	   System.out.println("end");
	}
}