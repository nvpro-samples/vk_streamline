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

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inWorldNormal;
layout(location = 2) in vec3 inWorldPosPrev;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outMotion;

layout(push_constant) uniform NodeInfo_
{
  NodeInfo nodeInfo;
};

layout(set = 0, binding = 0) uniform FrameInfo_
{
  FrameInfo frameInfo;
};

vec4 getColor(vec3 viewDir, vec3 normal)
{
  vec3 lightDir = viewDir;

  vec4 color = vec4(nodeInfo.color, 1.0);
  // Diffuse
  color.rgb *= abs(dot(normal, lightDir));
  // Specular
  color.rgb += pow(max(0.0, dot(normal, normalize(lightDir + viewDir))), 16.0);

  return color;
}

vec2 getMotion(vec3 worldPos, vec3 worldPosPrev)
{
    vec4 clipPos = frameInfo.viewProj * vec4(worldPos, 1.0);
    vec4 clipPosPrev = frameInfo.viewProjPrev * vec4(worldPosPrev, 1.0);

    return ((clipPosPrev.xy / clipPosPrev.w) - (clipPos.xy / clipPos.w)) * 0.5;
}

void main()
{
  outColor = getColor(normalize(frameInfo.camPos - inWorldPos), inWorldNormal);
  outMotion = getMotion(inWorldPos, inWorldPosPrev);
}
