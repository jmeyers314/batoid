all: jtrace.so libjtrace.dylib



jtrace.so: libjtrace.dylib pyjtrace.o pyvec3.o pyray.o pyintersection.o pysurface.o pyparaboloid.o pyasphere.o pyplane.o
	g++ -o jtrace.so -fvisibility=hidden -bundle pyjtrace.o pyvec3.o pyray.o pyintersection.o pysurface.o pyparaboloid.o pyasphere.o pyplane.o -L/Users/josh/src/lsstsw3/miniconda/lib -L/Users/josh/src/jtrace/ -lpython3.5m -ljtrace



pyjtrace.o: pysrc/jtrace.cpp include/jtrace.h
	g++ -o pyjtrace.o -c -std=c++11 -fvisibility=hidden -Iinclude/ -I/Users/josh/src/lsstsw3/miniconda/include/python3.5m/ pysrc/jtrace.cpp

pyvec3.o: pysrc/vec3.cpp include/vec3.h
	g++ -o pyvec3.o -c -std=c++11 -fvisibility=hidden -Iinclude/ -I/Users/josh/src/lsstsw3/miniconda/include/python3.5m/ pysrc/vec3.cpp

pyray.o: pysrc/ray.cpp include/ray.h
	g++ -o pyray.o -c -std=c++11 -fvisibility=hidden -Iinclude/ -I/Users/josh/src/lsstsw3/miniconda/include/python3.5m/ pysrc/ray.cpp

pyintersection.o: pysrc/intersection.cpp include/intersection.h
	g++ -o pyintersection.o -c -std=c++11 -fvisibility=hidden -Iinclude/ -I/Users/josh/src/lsstsw3/miniconda/include/python3.5m/ pysrc/intersection.cpp

pysurface.o: pysrc/surface.cpp include/surface.h
	g++ -o pysurface.o -c -std=c++11 -fvisibility=hidden -Iinclude/ -I/Users/josh/src/lsstsw3/miniconda/include/python3.5m/ pysrc/surface.cpp

pyparaboloid.o: pysrc/paraboloid.cpp include/paraboloid.h
	g++ -o pyparaboloid.o -c -std=c++11 -fvisibility=hidden -Iinclude/ -I/Users/josh/src/lsstsw3/miniconda/include/python3.5m/ pysrc/paraboloid.cpp

pyasphere.o: pysrc/asphere.cpp include/asphere.h
	g++ -o pyasphere.o -c -std=c++11 -fvisibility=hidden -Iinclude/ -I/Users/josh/src/lsstsw3/miniconda/include/python3.5m/ pysrc/asphere.cpp

pyplane.o: pysrc/plane.cpp include/plane.h
	g++ -o pyplane.o -c -std=c++11 -fvisibility=hidden -Iinclude/ -I/Users/josh/src/lsstsw3/miniconda/include/python3.5m/ pysrc/plane.cpp



libjtrace.dylib: intersection.o ray.o utils.o jtrace.o paraboloid.o asphere.o plane.o
	g++ -o libjtrace.dylib -dynamiclib ray.o utils.o jtrace.o intersection.o paraboloid.o asphere.o plane.o

jtrace.o: src/jtrace.cpp include/jtrace.h
	g++ -o jtrace.o -c -std=c++11 -I include/ src/jtrace.cpp

utils.o: src/utils.cpp include/utils.h
	g++ -o utils.o -c -std=c++11 -I include/ src/utils.cpp

ray.o: src/ray.cpp include/ray.h
	g++ -o ray.o -c -std=c++11 -I include/ src/ray.cpp

intersection.o: src/intersection.cpp include/intersection.h
	g++ -o intersection.o -c -std=c++11 -I include/ src/intersection.cpp

paraboloid.o: src/paraboloid.cpp include/paraboloid.h
	g++ -o paraboloid.o -c -std=c++11 -I include/ src/paraboloid.cpp

asphere.o: src/asphere.cpp include/asphere.h
	g++ -o asphere.o -c -std=c++11 -I include/ src/asphere.cpp

plane.o: src/plane.cpp include/plane.h
	g++ -o plane.o -c -std=c++11 -I include/ src/plane.cpp
