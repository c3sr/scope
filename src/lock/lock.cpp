#include <pthread.h>

int main() {
  pthread_mutex_t my_mutex;

  pthread_mutexattr_t my_attr;

  pthread_mutexattr_init(&my_attr);

  pthread_mutexattr_settype(&my_attr, PTHREAD_MUTEX_RECURSIVE); // Make the mutex reenterant

  pthread_mutex_init(&my_mutex, &my_attr);

  int i = 0;
  while (i < 50000000) { // Only one thread locks and unlocks the lock

    pthread_mutex_lock(&my_mutex);

    pthread_mutex_unlock(&my_mutex);

    i++;
  }
}
