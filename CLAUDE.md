# Notas para Claude

## Estado actual del proyecto

**Phase 0**: ✅ Completada
**Phase 1**: ✅ Completa - MVP funcional

## Lo que funciona

- ✅ Captura de audio con VAD (Voice Activity Detection)
- ✅ Transcripción con Whisper (OpenAI STT)
- ✅ Generación de respuestas con GPT-4o-mini (streaming)
- ✅ Síntesis de voz con OpenAI TTS (formato WAV, 24kHz)
- ✅ Reproducción de audio con cola secuencial (sounddevice)
- ✅ Lip-sync (boca se mueve con el audio)
- ✅ Avatar Pygame con cara programática y parpadeo
- ✅ Indicador de estado (verde=listening, amarillo=thinking, azul=speaking)
- ✅ Tests unitarios pasando (57 tests)
- ✅ mypy y ruff sin errores
- ✅ Latencia reducida con streaming híbrido

## Optimización de latencia implementada

El sistema usa **streaming híbrido** para reducir time-to-first-audio:

1. **Primer chunk (~80+ chars)** → TTS inmediato (baja latencia)
2. **Resto del texto** → TTS cuando termina el streaming (mejor prosodia)

### Flujo en dos fases:

```
Fase 1: Streaming (VAD pausado)
───────────────────────────────────────────────
Main Thread                 Background Thread
run_frame() loop            LLM stream
     │                           │
     │ ← volume callbacks ──── TTS primer chunk → enqueue
     │                           │
     │                         TTS resto → enqueue
     │                           │
     └── streaming_done ────────←┘

Fase 2: Playback (VAD pausado)
───────────────────────────────────────────────
Main Thread                 Playback Thread
run_frame() loop            reproduce cola
     │                           │
     │ ← volume callbacks ───────│
     │                           │
     └── playback_done ─────────←┘ (cola vacía)

Finally: Resume VAD (seguro, todo terminó)
```

## Arquitectura threading

```
Main Thread (Pygame)
    └── run_frame() loop continuo
    └── procesa speech_queue (audio del usuario)
    └── _stream_and_speak() coordina respuesta

Background Thread (LLM + TTS)
    └── stream_and_synthesize()
    └── LLM streaming → detecta chunks → TTS → encola

Playback Thread (AudioPlayback)
    └── Cola autónoma de reproducción
    └── Reproduce secuencialmente
    └── Callbacks de volumen para lip-sync

Audio Input Thread (VAD)
    └── sounddevice InputStream callback
    └── Detecta voz → pone en speech_queue
    └── Se pausa durante respuesta del asistente
```

## Código relevante

- `rostro/runtime/controller.py`:
  - `_stream_and_speak()` - Orquesta streaming + TTS en dos fases
  - `_extract_first_chunk()` - Extrae ~80+ chars para primer TTS

- `rostro/audio/playback.py`:
  - Cola autónoma con `queue.Queue`
  - `play()` encola, thread reproduce secuencialmente
  - Callbacks: `volume_callback`, `on_queue_empty`

## Configuración importante

- `config/default.yaml`: `sample_rate: 24000` (evita resampling, OpenAI TTS es 24kHz)
- `MIN_FIRST_CHUNK_CHARS = 80` en controller (tamaño mínimo primer chunk)

## Comandos útiles

```bash
# Ejecutar
source venv/bin/activate
python -m rostro.main

# Tests
pytest tests/unit/ -v

# Checks de calidad
ruff format . && ruff check . && mypy rostro/
```

## Debug logs

Los logs `[Stream]`, `[Playback]`, `[VAD]`, `[AudioCapture]` están activos. Quitar antes de release.
