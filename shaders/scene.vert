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

#version 450

#extension GL_GOOGLE_include_directive : enable

#include "common.h"

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outWorldNormal;
layout(location = 2) out vec3 outWorldPosPrev;

layout(push_constant) uniform NodeInfo_
{
  NodeInfo nodeInfo;
};

layout(set = 0, binding = 0) uniform FrameInfo_
{
  FrameInfo frameInfo;
};

void main()
{
  vec4 pos = nodeInfo.model * vec4(inPos, 1.0);
  gl_Position = frameInfo.viewProj * pos;

  // Apply jitter offset
  gl_Position.xy += frameInfo.jitterOffset * gl_Position.w;

  outWorldPos = pos.xyz;
  outWorldNormal = inNormal;

  vec4 posPrev = nodeInfo.modelPrev * vec4(inPos, 1.0);
  outWorldPosPrev = posPrev.xyz;
}
