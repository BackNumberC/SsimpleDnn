# SsimpleDnn
一个简单的DNN框架（ C++）

#运行环境
vs+opencv即可  只是调用了imread和resize（用于读取图片并转化为行向量输入）版本不需要太高

#代码内容
1.大概就是一个简版DNN（只有全连接层）框架，可通过函数接口自由设置lenrning-rate,batch-size,epoch,Data-size，以及中间层和输出层
2.训练及测试时读取txt存储的图片路径
3.可保存模型参数到xml文件，也可下次启动调用xml文件加载模型



