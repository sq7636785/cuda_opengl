#ifndef PATH_TRACER_H
#define PATH_TRACER_H

#include <vector>
#include "scene.h"

void pathTraceInit(Scene* scene);
void pathTraceFree();
void pathTrace(uchar4* pbo, int frame, int iteration);



#endif // !PATH_TRACER
