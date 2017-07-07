model=svmTrain(X,y,100,@(x1,x2) gaussianKernel(x1,x2,0.1));
visualizeBoundary(X,y,model);
%visualizeBoundary(Xval,yval,model);