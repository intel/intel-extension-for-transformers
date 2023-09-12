//  Copyright (c) 2021 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#ifndef ENGINE_SPARSELIB_SRC_SINGLETON_H_
#define ENGINE_SPARSELIB_SRC_SINGLETON_H_
#include <mutex>  // NOLINT

template <typename T>
class Singleton {
 public:
  static T* GetInstance() {
    if (instance_ == nullptr) {
      std::lock_guard<std::mutex> guard(mutex_);
      if (instance_ == nullptr) {
        instance_ = new (std::nothrow) T();
      }
    }
    return instance_;
  }

  static bool Destroy() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (instance_ != nullptr) {
      delete instance_;
      instance_ = nullptr;
      return true;
    }
    return false;
  }

 private:
  Singleton() = default;
  ~Singleton() = default;
  Singleton(const Singleton&) = delete;
  Singleton& operator=(const Singleton&) = delete;

 private:
  static T* instance_;
  static std::mutex mutex_;
};

template <typename T>
T* Singleton<T>::instance_ = nullptr;

template <typename T>
std::mutex Singleton<T>::mutex_;
#endif  // ENGINE_SPARSELIB_SRC_SINGLETON_H_
