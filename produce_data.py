from HW5lib import *
import csv

M = [100,200,300,400,500,600,700,800,900,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,8000,9000,10000] 


with open("data_overfitting.dat", "a") as f:
    writer = csv.writer(f)
    print("\n*******************\n")
    for m in M:
        print("Creation network m={}".format(m))
        net = mnist_classifier(m,imgs_val,lbls_val)
        print("Computing errors...")
        err_val = net.compute_error(imgs,lbls)
        err     = net.compute_error(imgs_val,lbls_val)
        print(err)
        print(err_val)
#        errs = []
#        for i,n in enumerate(net.nets):
#            prediction = n.predict(imgs_val)
#            prediction_bool = map(lambda x,y: x==y, np.sign(prediction),mnist_classifier.y(lbls_val,i).A1)
#            errs.append(100-sum(prediction_bool)/len(lbls_val)*100)
#                    
#        writer.writerow([m, err, err_val, *errs])
        writer.writerow([m, err, err_val])
        
        print("\n*******************\n")

