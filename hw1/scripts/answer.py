
import numpy as np
import pandas as pd
import save_csv
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
#导入数据
def load_data(path):
    data = np.load(path)
    X_train = data['training_data']
    y_train = data['training_labels']
    X_test = data['test_data']
    return X_train, y_train, X_test
#定义计算准确率的函数
def compute_accuracy(y_true, y_pred):
    current=0
    total=len(y_true)
    for i in range(total):
        if y_true[i]==y_pred[i]:
            current+=1
    accuracy=current/total
    return accuracy
#定义打乱数据的函数,提取训练集和验证集
def shuffle_data(X,y,ratio,seed):
    np.random.seed(seed)
    data_size=X.shape[0]
    index=np.arange(data_size)
    np.random.shuffle(index)
    split_point=int(data_size*(1-ratio))
    train_index,val_index=index[:split_point],index[split_point:]
    X_train,X_val=X[train_index],X[val_index]
    y_train,y_val=y[train_index],y[val_index]
    return X_train,X_val,y_train,y_val
#定义训练函数
def train_and_evaluate(X_train,X_val,y_train,y_val, train_sizes):
    train_accuracies = []
    val_accuracies=[]
    for size in train_sizes:
        # 获取训练集子集
        X_sub = X_train[:size]
        y_sub = y_train[:size]
        # 创建并训练 SVM 模型
        model = LinearSVC(max_iter=10000)
        model.fit(X_sub, y_sub)
        # 计算训练准确率（只用当前子集）
        y_pred_train = model.predict(X_sub)
        train_acc = compute_accuracy(y_sub, y_pred_train)
        train_accuracies.append(train_acc)
        # 计算验证准确率（用整个测试集）
        y_pred_val = model.predict(X_val)
        val_acc = compute_accuracy(y_val, y_pred_val)
        val_accuracies.append(val_acc)
    return train_accuracies, val_accuracies
#画图函数
def plot_results(train_sizes, train_accuracies, val_accuracies):
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_accuracies, marker='o', label='Training Accuracy')
    plt.plot(train_sizes, val_accuracies, marker='s', label='Validation Accuracy')
    plt.title('Accuracy vs Number of Training Examples')
    plt.xlabel('Number of Training Examples')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
#图像矩阵降维，归一化
def data_deal(X):
    X_dealed=X.reshape(X.shape[0],-1)
    X_dealed = X_dealed.astype(np.float32) / 255.0
    return X_dealed
def hyperparameter(X_train,y_train, X_val, y_val):
    C_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]  
    results = []
    for c in C_values:
        model=LinearSVC(C=c,max_iter=10000)
        model.fit(X_train,y_train)
        y_predict=model.predict(X_val)
        val_accuracies=compute_accuracy(y_val,y_predict)
        results.append((c,val_accuracies))
        print(f"C={c}, Validation Accuracy={val_accuracies:.4f}")
    best_C, best_accuracies = max(results, key=lambda x: x[1])
    print(f"best_C={c},  best_accuracies={ best_accuracies:.4f}")
    return best_C
#交叉验证
def cross_val(X_train,y_train,c_values):
    np.random.seed(60)
    train_size=X_train.shape[0]
    index=np.arange(train_size)
    np.random.shuffle(index)
    X_train,y_train=X_train[index],y_train[index]
    fold_size=train_size//5
    result=[]
    for c in c_values:
        aver_acc=0
        for i in range(5):
            start=i*fold_size
            end=(i+1)*fold_size if i!=4 else train_size
            X_train_fold = np.concatenate((X_train[:start], X_train[end:]), axis=0)
            y_train_fold = np.concatenate((y_train[:start], y_train[end:]), axis=0)
            X_val=X_train[start:end]
            y_val=y_train[start:end]
            model=LinearSVC(C=c,max_iter=10000)
            model.fit(X_train_fold,y_train_fold)
            y_pred=model.predict(X_val)
            acc=compute_accuracy(y_val,y_pred)
            aver_acc+=acc
        aver_acc=aver_acc/5
        result.append((aver_acc,c))
        print(f'c={c},accuracy={aver_acc}')
    best_aver_acc,best_c=max(result,key=lambda x:x[0])
    print(f'best c={best_c},best accuracy={best_aver_acc}')
    return best_c
def results_to_csv(y_test,name):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1 # Ensures that the index starts at 1
    df.to_csv(name, index_label='Id')




if __name__=='__main__':
    #处理spam文件
    path_spam='../data/spam-data.npz'
    X_spam, y_spam, X_test_spam=load_data(path_spam)
    X_train_spam,X_val_spam,y_train_spam,y_val_spam=shuffle_data(X_spam,y_spam,0.2,50)
    train_size_spam=[ 100, 200, 500, 1000, 2000, len(X_train_spam)]
    train_accuracies_sapm, val_accuracies_spam=train_and_evaluate(X_train_spam,X_val_spam,y_train_spam,y_val_spam,train_size_spam)
    plot_results(train_size_spam,train_accuracies_sapm, val_accuracies_spam)
    #处理mnist文件
    path_mnist='../data/mnist-data.npz'
    X_mnist, y_mnist, X_test_mnist=load_data(path_mnist)
    X_mnist, X_test_mnist=data_deal(X_mnist),data_deal(X_test_mnist)
    X_train_mnist,X_val_mnist,y_train_mnist,y_val_mnist=shuffle_data(X_mnist,y_mnist,10000/len(X_mnist),50)
    train_size_mnist=[ 100, 200, 500, 1000, 2000, 5000, 10000]
    train_accuracies_mnist, val_accuracies_mnist=train_and_evaluate(X_train_mnist,X_val_mnist,y_train_mnist,y_val_mnist,train_size_mnist)
    plot_results(train_size_mnist,train_accuracies_mnist, val_accuracies_mnist)
    #手动寻找超参数
    print('=======c_mnist数据集参数搜索=======')
    c_mnist=hyperparameter(X_train_mnist,y_train_mnist,X_val_mnist,y_val_mnist)
    print('=======spam数据集参数搜索(5折交叉检验)=======')
    C_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]  
    c_spam=cross_val(X_spam,y_spam,C_values)
    #用最优超参数预测
    model=LinearSVC(C=c_mnist,max_iter=10000)
    model.fit(X_mnist,y_mnist)
    y_pred_minst=model.predict(X_test_mnist)
    results_to_csv(y_pred_minst,'my_submission_minst.csv')
    model=LinearSVC(C=c_spam,max_iter=10000)
    model.fit(X_spam,y_spam)
    y_pred_spam=model.predict(X_test_spam)
    results_to_csv(y_pred_spam,'my_submission_spam.csv')


# %%
