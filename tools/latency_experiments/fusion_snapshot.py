import json,time,ntcore
n=ntcore.NetworkTableInstance.create();n.startClient4('highrate-validation');n.setServer('127.0.0.1')
keys=[f'HighRate/{lane}/{metric}' for lane in ['main20','thread5'] for metric in ['Updates','Accepted','ErrorMeters','MaxGapMs']]
s={k:n.getDoubleTopic('/SmartDashboard/'+k).subscribe(-1) for k in keys}
time.sleep(2);a={k:v.get() for k,v in s.items()};time.sleep(3);b={k:v.get() for k,v in s.items()}
print(json.dumps({'connected':n.isConnected(),'start':a,'end':b},indent=2));n.stopClient()
