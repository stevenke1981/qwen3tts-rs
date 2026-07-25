#ifndef QWEN3TTS_H
#define QWEN3TTS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qwen3tts_model qwen3tts_model;
typedef struct qwen3tts_session qwen3tts_session;

typedef void (*qwen3tts_audio_callback)(
    const float * samples,
    size_t sample_count,
    uint32_t sample_rate,
    int is_final,
    void * user_data
);

int qwen3tts_model_load(const char * model_path, qwen3tts_model ** out_model);
void qwen3tts_model_free(qwen3tts_model * model);

int qwen3tts_session_create(
    qwen3tts_model * model,
    const char * request_json,
    qwen3tts_audio_callback callback,
    void * user_data,
    qwen3tts_session ** out_session
);
int qwen3tts_session_run(qwen3tts_session * session);
void qwen3tts_session_cancel(qwen3tts_session * session);
void qwen3tts_session_free(qwen3tts_session * session);

const char * qwen3tts_last_error(void);

#ifdef __cplusplus
}
#endif
#endif
