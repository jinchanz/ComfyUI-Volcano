`POST https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks`   [ ](https://api.volcengine.com/api-explorer/?action=CreateContentsGenerationsTasks&data=%7B%7D&groupName=%E8%A7%86%E9%A2%91%E7%94%9F%E6%88%90API&query=%7B%7D&serviceCode=ark&version=2024-01-01)[运行](https://api.volcengine.com/api-explorer/?action=CreateContentsGenerationsTasks&data=%7B%7D&groupName=%E8%A7%86%E9%A2%91%E7%94%9F%E6%88%90API&query=%7B%7D&serviceCode=ark&version=2024-01-01)
本文介绍创建视频生成任务 API 的输入输出参数，供您使用接口时查阅字段含义。模型会依据传入的图片及文本信息生成视频，待生成完成后，您可以按条件查询任务并获取生成的视频。

**不同模型支持的视频生成能力简介**

* **Seedance 1.5 pro==^new^==** ** ** **==^有声视频^==** **(自定义是否包含音频)** 
   * 图生视频\-首尾帧，根据您输入的++首帧图片+尾帧图片+文本提示词（可选）+参数（可选）++ 生成目标视频。
   * 图生视频\-首帧，根据您输入的++首帧图片+文本提示词（可选）+参数（可选）++ 生成目标视频。
   * 文生视频，根据您输入的++文本提示词+参数（可选）++ 生成目标视频。
* **Seedance 1.0 pro**
   * 图生视频\-首尾帧，根据您输入的++首帧图片+尾帧图片+文本提示词（可选）+参数（可选）++ 生成目标视频。
   * 图生视频\-首帧，根据您输入的++首帧图片+文本提示词（可选）+参数（可选）++ 生成目标视频。
   * 文生视频，根据您输入的++文本提示词+参数（可选）++ 生成目标视频。
* **Seedance 1.0 pro fast**
   * 图生视频\-首帧，根据您输入的++首帧图片+文本提示词（可选）+参数（可选）++ 生成目标视频。
   * 文生视频，根据您输入的++文本提示词+参数（可选）++ 生成目标视频。
* **Seedance 1.0 lite**
   * **doubao\-seedance\-1\-0\-lite\-t2v：** 文生视频，根据您输入的++文本提示词+参数（可选）++ 生成目标视频。
   * **doubao\-seedance\-1\-0\-lite\-i2v：** 
      * 图生视频\-参考图，根据您输入的**++参考图片（1\-4张）++ **  +++文本提示词（可选）+ 参数（可选）++ 生成目标视频。
      * 图生视频\-首尾帧，根据您输入的++首帧图片+尾帧图片+文本提示词（可选）+参数（可选）++ 生成目标视频。
      * 图生视频\-首帧，根据您输入的++首帧图片+文本提示词（可选）+参数（可选）++ 生成目标视频。


Tips：一键展开折叠，快速检索内容
打开页面右上角开关，**ctrl ** + **f** 可检索页面内所有内容。
<span>![图片](https://portal.volccdn.com/obj/volcfe/cloud-universal-doc/upload_cae7ddb0e1977b68b353f17897b8574c.png) </span>


```mixin-react
return (<Tabs>
<Tabs.TabPane title="在线调试" key="cKmdyIjR"><RenderMd content={`<APILink link="https://api.volcengine.com/api-explorer/?action=CreateContentsGenerationsTasks&data=%7B%7D&groupName=%E8%A7%86%E9%A2%91%E7%94%9F%E6%88%90API&query=%7B%7D&serviceCode=ark&version=2024-01-01" description="API Explorer 您可以通过 API Explorer 在线发起调用，无需关注签名生成过程，快速获取调用结果。"></APILink>
`}></RenderMd></Tabs.TabPane>
<Tabs.TabPane title="鉴权说明" key="vRJT6oJZ"><RenderMd content={`本接口仅支持 API Key 鉴权，请在 [获取 API Key](https://console.volcengine.com/ark/region:ark+cn-beijing/apiKey) 页面，获取长效 API Key。
`}></RenderMd></Tabs.TabPane>
<Tabs.TabPane title="快速入口" key="MlbBRTbjal"><RenderMd content={` [ ](#)[体验中心](https://console.volcengine.com/ark/region:ark+cn-beijing/experience/vision)       <span>![图片](https://portal.volccdn.com/obj/volcfe/cloud-universal-doc/upload_2abecd05ca2779567c6d32f0ddc7874d.png =20x) </span>[模型列表](https://www.volcengine.com/docs/82379/1330310?lang=zh#2705b333)       <span>![图片](https://portal.volccdn.com/obj/volcfe/cloud-universal-doc/upload_a5fdd3028d35cc512a10bd71b982b6eb.png =20x) </span>[模型计费](https://www.volcengine.com/docs/82379/1544106?redirect=1&lang=zh#02affcb8)       <span>![图片](https://portal.volccdn.com/obj/volcfe/cloud-universal-doc/upload_afbcf38bdec05c05089d5de5c3fd8fc8.png =20x) </span>[API Key](https://console.volcengine.com/ark/region:ark+cn-beijing/apiKey?apikey=%7B%7D)
 <span>![图片](https://portal.volccdn.com/obj/volcfe/cloud-universal-doc/upload_57d0bca8e0d122ab1191b40101b5df75.png =20x) </span>[调用教程](https://www.volcengine.com/docs/82379/1366799)       <span>![图片](https://portal.volccdn.com/obj/volcfe/cloud-universal-doc/upload_f45b5cd5863d1eed3bc3c81b9af54407.png =20x) </span>[接口文档](https://www.volcengine.com/docs/82379/1520758)       <span>![图片](https://portal.volccdn.com/obj/volcfe/cloud-universal-doc/upload_1609c71a747f84df24be1e6421ce58f0.png =20x) </span>[常见问题](https://www.volcengine.com/docs/82379/1359411)       <span>![图片](https://portal.volccdn.com/obj/volcfe/cloud-universal-doc/upload_bef4bc3de3535ee19d0c5d6c37b0ffdd.png =20x) </span>[开通模型](https://console.volcengine.com/ark/region:ark+cn-beijing/openManagement?LLM=%7B%7D&OpenTokenDrawer=false)
`}></RenderMd></Tabs.TabPane></Tabs>);
```


---


<span id="RxN8G2nH"></span>
## 请求参数 
> 跳转 [响应参数](#y2hhTyHB)

<span id="BJ5XLFqM"></span>
### 请求体

---


**model** `string` %%require%%
您需要调用的模型的 ID （Model ID），[开通模型服务](https://console.volcengine.com/ark/region:ark+cn-beijing/openManagement?LLM=%7B%7D&OpenTokenDrawer=false)，并[查询 Model ID](https://www.volcengine.com/docs/82379/1330310) 。
您也可通过 Endpoint ID 来调用模型，获得限流、计费类型（前付费/后付费）、运行状态查询、监控、安全等高级能力，可参考[获取 Endpoint ID](https://www.volcengine.com/docs/82379/1099522)。

---


**content** `object[]` %%require%%
输入给模型，生成视频的信息，支持文本信息和图片信息。

信息类型

---


**文本信息** `object`
输入给模型生成视频的内容，文本内容部分。

属性

---


content.**type ** `string` %%require%%
输入内容的类型，此处应为 `text`。

---


content.**text ** `string` %%require%%
输入给模型的文本内容，描述期望生成的视频，包括：

* **文本提示词（必填）** ：支持中英文。建议不超过500字。字数过多信息容易分散，模型可能因此忽略细节，只关注重点，造成视频缺失部分元素。提示词的更多使用技巧请参见 [Seedance 提示词指南](https://www.volcengine.com/docs/82379/1587797)。
* **参数（选填）** ：在文本提示词后追加\-\-[parameters]，控制视频输出的规格，详情见 **模型文本命令(选填** **）** 。


---


**图片信息** `object`
输入给模型生成视频的内容，图片信息部分。

属性

---


content.**type ** `string` %%require%%
输入内容的类型，此处应为 `image_url`。支持图片URL或图片 Base64 编码。

---


content.**image_url ** `object` %%require%%
输入给模型的图片对象。

属性

---


content.image_url.**url ** `string` %%require%%
图片信息，可以是图片URL或图片 Base64 编码。

* 图片URL：请确保图片URL可被访问。
* Base64编码：请遵循此格式`data:image/<图片格式>;base64,<Base64编码>`，注意 `<图片格式>` 需小写，如 `data:image/png;base64,{base64_image}`。

:::tip
传入图片需要满足以下条件：

* 图片格式：jpeg、png、webp、bmp、tiff、gif。其中，Seedance 1.5 pro 新增支持 heic 和 heif。
* 宽高比（宽/高）： (0.4, 2.5) 
* 宽高长度（px）：(300, 6000)
* 大小：小于 30 MB

:::

---


content.**role ** `string` `条件必填`
图片的位置或用途。
:::warning
首帧图生视频、首尾帧图生视频、参考图生视频为 3 种互斥的场景，不支持混用。

:::
图生视频\-首帧

* **支持模型：** 所有图生视频模型
* **字段role取值：** 需要传入1个image_url对象，且字段role可不填，或字段role为：first_frame


图生视频\-首尾帧

* **支持模型：** Seedance 1.5 pro、Seedance 1.0 pro、Seedance 1.0 lite i2v 
* **字段role取值：** 需要传入2个image_url对象，且字段role必填。
   * 首帧图片对应的字段role为：first_frame
   * 尾帧图片对应的字段role为：last_frame

:::tip
传入的首尾帧图片可相同。首尾帧图片的宽高比不一致时，以首帧图片为主，尾帧图片会自动裁剪适配。

:::

图生视频\-参考图

* **支持模型：** Seedance 1.0 lite i2v
* **字段role取值：** 需要传入1～4个image_url对象，且字段role必填。
   * 每张参考图片对应的字段role均为：reference_image

:::tip
参考图生视频功能的文本提示词，可以用自然语言指定多张图片的组合。但若想有更好的指令遵循效果，**推荐使用“[图1]xxx，[图2]xxx”的方式来指定图片**。
示例1：戴着眼镜穿着蓝色T恤的男生和柯基小狗，坐在草坪上，3D卡通风格
示例2：[图1]戴着眼镜穿着蓝色T恤的男生和[图2]的柯基小狗，坐在[图3]的草坪上，3D卡通风格

:::



---


**callback_url** `string` 
填写本次生成任务结果的回调通知地址。当视频生成任务有状态变化时，方舟将向此地址推送 POST 请求。
回调请求内容结构与[查询任务API](https://www.volcengine.com/docs/82379/1521309)的返回体一致。
回调返回的 status 包括以下状态：

* queued：排队中。
* running：任务运行中。
* succeeded： 任务成功。（如发送失败，即5秒内没有接收到成功发送的信息，回调三次）
* failed：任务失败。（如发送失败，即5秒内没有接收到成功发送的信息，回调三次）
* expired：任务超时，即任务处于**运行中或排队中**状态超过过期时间。可通过 **execution_expires_after ** 字段设置过期时间。


---


**return_last_frame** `boolean` `默认值 false`

* true：返回生成视频的尾帧图像。设置为 `true` 后，可通过 [查询视频生成任务接口](https://www.volcengine.com/docs/82379/1521309) 获取视频的尾帧图像。尾帧图像的格式为 png，宽高像素值与生成的视频保持一致，无水印。
   使用该参数可实现生成多个连续视频：以上一个生成视频的尾帧作为下一个视频任务的首帧，快速生成多个连续视频，调用示例详见 [教程](https://www.volcengine.com/docs/82379/1366799?lang=zh#141cf7fa)。
* false：不返回生成视频的尾帧图像。


---


**service_tier==^new^==** `string` `默认值 default`
> 不支持修改已提交任务的服务等级

指定处理本次请求的服务等级类型，枚举值：

* default：在线推理模式，RPM 和并发数配额较低（详见 [模型列表](https://www.volcengine.com/docs/82379/1330310?lang=zh#2705b333)），适合对推理时效性要求较高的场景。
* flex：离线推理模式，TPD 配额更高（详见 [模型列表](https://www.volcengine.com/docs/82379/1330310?lang=zh#2705b333)），价格为在线推理的 50%， 适合对推理时延要求不高的场景。


---


**execution_expires_after==^new^==** ** ** `integer` `默认值 172800`
任务超时阈值。指定任务提交后的过期时间（单位：秒），从 **created at** 时间戳开始计算。默认值 172800 秒，即 48 小时。取值范围：[3600，259200]。
不论使用哪种 **service_tier**，都建议根据业务场景设置合适的超时时间。超过该时间后任务会被自动终止，并标记为`expired`状态。

---


**generate_audio==^new^==** ** ** `boolean` `默认值 true`
> 仅 Seedance 1.5 pro 支持

控制生成的视频是否包含与画面同步的声音。

* true：模型输出的视频包含同步音频。Seedance 1.5 pro 能够基于文本提示词与视觉内容，自动生成与之匹配的人声、音效及背景音乐。建议将对话部分置于双引号内，以优化音频生成效果。例如：男人叫住女人说：“你记住，以后不可以用手指指月亮。”
* false：模型输出的视频为无声视频。

&nbsp;
<span id="L9tzcCyD"></span>
### 模型文本命令(选填)
**在文本提示词后追加 \-\-[parameters] ** ，控制视频输出的规格，包括宽高比、帧率、分辨率等。
> 不同模型，可能对应支持不同的参数与取值，详见 [输出视频格式](https://www.volcengine.com/docs/82379/1366799?lang=zh#9fe4cce0) 。当输入的参数或取值不符合所选的模型时，内容会被忽略或报错。


示例
```JSON
// 指定生成视频的宽高比为16:9，时长为 5 秒，帧率为 24 fps，分辨率为720p，包含水印，种子整数为11，不固定摄像头

"content": [
        {
            "type": "text",
            "text": "小猫对着镜头打哈欠。 --rs 720p --rt 16:9 --dur 5 --fps 24 --wm true --seed 11 --cf false"
        }
    ]

```




---


**resolution **  `string` `简写 rs`
> Seedance 1.5 pro、Seedance 1.0 lite 默认值：`720p`
> Seedance 1.0 pro&pro\-fast 默认值：`1080p`

视频分辨率，枚举值：

* 480p
* 720p
* 1080p：参考图场景不支持，Seedance 1.5 pro 暂不支持


---


**ratio ** `string` `简写 rt`
> 文生视频默认值一般是`16:9`
> 图生视频默认值一般是`adaptive`。特别注意，参考图生视频的默认值是`16:9`
> Seedance 1.5 pro 默认值是`adaptive`

生成视频的宽高比例。不同宽高比对应的宽高像素值见下方表格。

* 16:9 
* 4:3
* 1:1
* 3:4
* 9:16
* 21:9
* adaptive：根据所上传首帧图片的比例，自动选择最合适的宽高比。

:::warning
Seedance 1.5 pro 文生视频设置`adaptive`时，表示根据用户提示词，智能选择最合适的宽高比。实际生成视频的宽高比可通过 [查询视频生成任务 API](https://www.volcengine.com/docs/82379/1521309?lang=zh) 返回的 **ratio** 字段获取。

:::
**不同宽高比对应的宽高像素值**
Note：图生视频，选择的宽高比与您上传的图片宽高比不一致时，方舟会对您的图片进行裁剪，裁剪时会居中裁剪，详细规则见 [图片裁剪规则](https://www.volcengine.com/docs/82379/1366799?lang=zh#f76aafc8)。

|分辨率 |宽高比|宽高像素值|宽高像素值|\
| | |Seedance 1.0 系列 |Seedance 1.5 pro |
|---|---|---|---|
|480p |16:9 |864×480 |864×496 |
|^^|4:3 |736×544 |752×560 |
|^^|1:1 |640×640 |640×640 |
|^^|3:4 |544×736 |560×752 |
|^^|9:16 |480×864 |496×864 |
|^^|21:9 |960×416 |992×432 |
|720p |16:9 |1248×704 |1280×720 |
|^^|4:3 |1120×832 |1112×834 |
|^^|1:1 |960×960 |960×960 |
|^^|3:4 |832×1120 |834×1112 |
|^^|9:16 |704×1248 |720×1280 |
|^^|21:9 |1504×640 |1470×630 |
|1080p |16:9 |1920×1088 |\- |\
|> 1.5 pro 暂不支持| | | |\
|> 1.0 lite 参考图场景不支持 | | | |
|^^|4:3 |1664×1248 |\- |
|^^|1:1 |1440×1440 |\- |
|^^|3:4 |1248×1664 |\- |
|^^|9:16 |1088×1920 |\- |
|^^|21:9 |2176×928 |\- |




---


**duration** `integer` `默认值 5` `简写 dur`
> duration 和 frames 二选一即可，frames 的优先级高于 duration。如果您希望生成整数秒的视频，建议指定 duration。

生成视频时长，单位：秒。支持 2~12 秒。
:::warning
Seedance 1.5 pro 支持两种配置方法

   * 指定具体时长：支持 [4,12] 范围内的任一整数。
   * 不指定具体生成时长：设置为 `-1`，表示由模型在 [4,12] 范围内自主选择合适的视频长度（整数秒）。实际生成视频的时长可通过 [查询视频生成任务 API](https://www.volcengine.com/docs/82379/1521309?lang=zh) 返回的 **duration** 字段获取。注意视频时长与计费相关，请谨慎设置。


:::
---


**frames** `integer` `简写 frames`
> Seedance 1.5 pro 暂不支持
> duration 和 frames 二选一即可，frames 的优先级高于 duration。如果您希望生成小数秒的视频，建议指定 frames。

生成视频的帧数。通过指定帧数，可以灵活控制生成视频的长度，生成小数秒的视频。
由于 frames 的取值限制，仅能支持有限小数秒，您需要根据公式推算最接近的帧数。

* 计算公式：帧数 = 时长 × 帧率（24）。
* 取值范围：支持 [29, 289] 区间内所有满足 `25 + 4n` 格式的整数值，其中 n 为正整数。

例如：假设需要生成 2.4 秒的视频，帧数=2.4×24=57.6。由于 frames 不支持 57.6，此时您只能选择一个最接近的值。根据 25+4n 计算出最接近的帧数为 57，实际生成的视频为 57/24=2.375 秒。

---


**framespersecond**  `integer` `默认值 24` `简写 fps`
帧率，即一秒时间内视频画面数量。仅支持 24

---


**seed** `integer` `默认值 -1` `简写 seed`
种子整数，用于控制生成内容的随机性。
取值范围：[\-1, 2^32\-1]之间的整数。
:::warning

* 相同的请求下，模型收到不同的seed值，如：不指定seed值或令seed取值为\-1（会使用随机数替代）、或手动变更seed值，将生成不同的结果。
* 相同的请求下，模型收到相同的seed值，会生成类似的结果，但不保证完全一致。


:::
---


**camerafixed** `boolean` `默认值 false` `简写 cf`
> 参考图场景不支持

是否固定摄像头。枚举值：

* true：固定摄像头。平台会在用户提示词中追加固定摄像头，实际效果不保证。
* false：不固定摄像头。


---


**watermark** `boolean` `默认值 false` `简写 wm`
生成视频是否包含水印。枚举值：

* false：不含水印。
* true：含有水印。


---


<span id="y2hhTyHB"></span>
## 响应参数
> 跳转 [请求参数](#RxN8G2nH)

**id ** `string`
视频生成任务 ID 。创建视频生成任务为异步接口，获取 ID 后，需要通过 [查询视频生成任务 API](https://www.volcengine.com/docs/82379/1521309) 来查询视频生成任务的状态。任务成功后，会输出生成视频的`video_url`。


