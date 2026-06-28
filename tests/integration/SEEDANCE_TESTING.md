# Seedance Video Generation Integration Tests

This directory contains integration tests for the Seedance video generation node, including comprehensive tests for Seedance 2.0 features.

## Prerequisites

1. **Local Griptape Cloud proxy running** at `http://localhost:8000`
2. **ServiceModelConfig and ServiceModelConfigAuthDetails** configured in the local database:
   - Provider: BYTEPLUS_ARK
   - Model IDs: `dreamina-seedance-2-0-260128`, `dreamina-seedance-2-0-fast-260128`
   - API key configured

## Test Files

### `test_seedance_video_generation.py`

Basic integration test for all Seedance models.

**Usage:**

```bash
# Test default model (Seedance 2.0 Fast)
python tests/integration/test_seedance_video_generation.py

# Test with custom prompt
python tests/integration/test_seedance_video_generation.py --prompt "A cat walking"

# Test a specific model
python tests/integration/test_seedance_video_generation.py --model "Seedance 2.0"

# Test all models
python tests/integration/test_seedance_video_generation.py --test-all

# Test with GTC storage backend
python tests/integration/test_seedance_video_generation.py --storage-backend gtc
```

### `test_seedance_2_0_features.py`

Comprehensive tests for Seedance 2.0 specific features.

**Usage:**

```bash
# Run all 2.0 feature tests
python tests/integration/test_seedance_2_0_features.py --test all

# Test basic text-to-video
python tests/integration/test_seedance_2_0_features.py --test text-to-video

# Test Seedance 2.0 Fast
python tests/integration/test_seedance_2_0_features.py --test fast

# Test audio generation
python tests/integration/test_seedance_2_0_features.py --test audio

# Test smart duration (-1)
python tests/integration/test_seedance_2_0_features.py --test smart-duration

# Test with GTC storage backend
python tests/integration/test_seedance_2_0_features.py --storage-backend gtc --test all
```

## Test Coverage

### Seedance 2.0 Standard
- ✓ Text-to-video generation
- ✓ 720p resolution
- ✓ Duration 4-15 seconds
- ✓ Smart duration (-1)
- ✓ Audio generation
- ✓ Adaptive aspect ratio

### Seedance 2.0 Fast
- ✓ Text-to-video generation
- ✓ 480p/720p resolution
- ✓ Duration 4-15 seconds
- ✓ Smart duration (-1)
- ✓ Audio generation
- ✓ Faster inference

### Feature Validation
- ✓ Parameter visibility based on model selection
- ✓ 2.0 models hide `camera_fixed` parameter
- ✓ 2.0 models show video/audio inputs
- ✓ Duration range validation (4-15 or -1 for 2.0)
- ✓ Resolution validation (no 1080p for 2.0)

### Future Tests (TODO)
- [ ] Reference images (multimodal, 0-9 images)
- [ ] Reference videos (0-3 videos)
- [ ] Reference audio (0-3 audio files)
- [ ] First/last frame image-to-video
- [ ] Validation: multimodal refs cannot mix with first/last frame
- [ ] Validation: audio requires image/video

## Debugging

If tests fail, check:

1. **Local proxy is running:**
   ```bash
   cd ~/code/griptape-cloud
   make up
   ```

2. **Database has ServiceModelConfig records:**
   ```bash
   # Access Django shell
   docker exec -it griptape-cloud-web-1 python manage.py shell_plus

   # Check models
   >>> ServiceModelConfig.objects.filter(model_name__contains='seedance-2-0')
   ```

3. **API key is valid:**
   - Check ServiceModelConfigAuthDetails in Django admin
   - Verify BytePlus API key is valid and activated

4. **Review logs:**
   ```bash
   docker logs -f griptape-cloud-web-1
   docker logs -f griptape-cloud-worker-1
   ```

## Expected Output

Successful test output:
```
============================================================
Test: Seedance 2.0 Fast Text-to-Video
============================================================
✓ Seedance 2.0 Fast text-to-video test passed
Output: {'End Flow': {'result': 'Video generated successfully and saved as seedance_video.mp4.'}}
```

Failed test output:
```
✗ Seedance 2.0 text-to-video test failed: Generation failed with status: failed
```

## Integration with CI/CD

To run these tests in CI:

```bash
# Start local proxy
make up

# Wait for services to be ready
sleep 10

# Run tests
python tests/integration/test_seedance_2_0_features.py --test all --storage-backend local

# Cleanup
make down
```

## Notes

- Tests use the local storage backend by default
- Generated videos are saved to the project file path or temp directory
- Each test creates a new workflow to ensure isolation
- Tests validate both successful generation and proper error handling
- 2.0 models default to 720p resolution for optimal quality/speed balance
