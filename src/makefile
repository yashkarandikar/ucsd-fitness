FLAGS=-O2
INCLUDE=/home/yashodhan/software/dlib-18.14
all:
	g++ -Wall -std=c++11 $(FLAGS) predictor_duration_evolving_user_cpp.cpp -llbfgs -o predictor_duration_evolving_user_cpp
	g++ -Wall -std=c++11 -I$(INCLUDE) $(FLAGS) predictor_duration_user_cpp.cpp -llbfgs -o predictor_duration_user_cpp
	#g++ -Wall lbfgs_example.c -llbfgs -o lbfgs_example
	#g++ -Wall -I$(INCLUDE) dlib_example.cpp -o dlib_example
	#g++ -Wall -I$(INCLUDE) dlib_example2.cpp -o dlib_example2
clean:
	rm -f predictor_duration_evolving_user_cpp predictor_duration_user_cpp
