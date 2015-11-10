import numpy

def get_num_correct(W,U,hbias,outbias,testx,testy,n_samples,n_output):
	print n_samples
	num_correct = 0

	for j in range(0,n_samples):
# print j
		y_proc = numpy.zeros(n_output)
		for i in range(0,n_output):
			activation_term = numpy.dot(testx[j,],W) + hbias + numpy.transpose(U)[i,]
			expterm = numpy.exp(activation_term)
			sumterm = expterm + 1
			producterm = numpy.prod(sumterm)
# y_proc = T.set_subtensor(y_proc[i,y[i]-1], a)
			y_proc[i] = producterm * numpy.exp(outbias[i])
		# print numpy.argmax(y_proc), testy[j]
		if(numpy.argmax(y_proc) == testy[j]):
			num_correct+=1
	return num_correct


 #        print j
 #        if(numpy.argmax(y_proc) == testy[j]):
 #        	num_correct+=1
	# return num_correct





	# 	if(numpy.argmax(y_proc) == testy[j]):
	# 		num_correct+=1

	# return num_correct
 #    # return T.sum(T.neq(T.argmax(y_proc,axis=0)+numberOne,self.testy))