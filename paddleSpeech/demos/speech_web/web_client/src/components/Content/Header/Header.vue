<template>
<div className="speech_header">
      <div className="speech_header_title">
        OceanSpeech
      </div>
      <div className="speech_header_describe">
         OceanSpeech是基于飞桨-PaddleSpeech封装的个人语音助手。  支持语音识别，语音合成，声纹识别，声音分类，语音唤醒，语音翻译等多种语音任务。
      </div>
      <div className="speech_header_link_box">
        <a href="https://github.com/OcaenEyes" className="speech_header_link"  target='_blank' rel='noreferrer' key={index}>
            前往Github
        </a>
      </div>
    </div>

</template>

<script>
export default {
    name:"Header"
}
</script>

<style lang="less" scoped>
@import "./style.less";
</style>