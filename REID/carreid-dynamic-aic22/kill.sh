ps -ef|grep car_resnet50 |grep -v grep | awk '{print "kill -9 "$2}' | sh
