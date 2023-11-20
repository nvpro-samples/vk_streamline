/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifdef __cplusplus
using mat4 = glm::mat4;
using vec2 = glm::vec2;
using vec3 = glm::vec3;
#endif  // __cplusplus

struct NodeInfo
{
  mat4 model;
  mat4 modelPrev;
  vec3 color;
};

struct FrameInfo
{
  mat4 viewProj;
  mat4 viewProjPrev;
  vec2 jitterOffset;
  vec3 camPos;
};
