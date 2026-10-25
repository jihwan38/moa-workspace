use super::*;
use crate::host::{audio::AudioFrames, image::ImagePatches, model::Input};

fn inputs<'a>(
    count: usize,
    image: Option<&'a ImagePatches>,
    audio: Option<&'a AudioFrames>,
) -> Vec<Input<'a>> {
    (0..count)
        .map(|row| Input {
            token: row + 17,
            soft: None,
            logits: row + 1 == count,
            image,
            audio,
        })
        .collect()
}

fn image() -> ImagePatches {
    ImagePatches {
        pixels: vec![
            vec![bf16::from_f32(0.25); Rv::SIZE],
            vec![bf16::from_f32(-0.5); Rv::SIZE],
        ],
        positions: vec![(3, 7), (usize::MAX, 1)],
    }
}

fn audio() -> AudioFrames {
    AudioFrames {
        frames: vec![
            vec![bf16::from_f32(0.125); Aa::SIZE],
            vec![bf16::from_f32(-0.75); Aa::SIZE],
        ],
    }
}

#[test]
fn mixed_media_keeps_target_rows_order_duplicates_and_original_positions() {
    let image = image();
    let audio = audio();
    let mut tokens = inputs(P128::SIZE, Some(&image), Some(&audio));
    tokens[15].soft = Some(SoftToken::Image(1));
    tokens[16].soft = Some(SoftToken::Audio(1));
    tokens[63].soft = Some(SoftToken::Image(0));
    tokens[64].soft = Some(SoftToken::Audio(0));
    tokens[95].soft = Some(SoftToken::Audio(1));
    tokens[127].soft = Some(SoftToken::Image(1));
    let blocks = runtime::Blocks::default();
    let sequence = Sequence { inputs: &tokens, start_position: 384, blocks: &blocks };
    let (images, frames) = collect_media_rows(&sequence).unwrap();

    assert_eq!(images.iter().map(|r| r.0).collect::<Vec<_>>(), [15, 63, 127]);
    assert_eq!(frames.iter().map(|r| r.0).collect::<Vec<_>>(), [16, 64, 95]);
    for (entry, index) in images.iter().zip([1, 0, 1]) {
        assert_eq!(entry.1, image.pixels[index].as_slice());
        assert_eq!(entry.2, image.positions[index]);
        assert_eq!(entry.1.as_ptr(), image.pixels[index].as_ptr());
    }
    for (entry, index) in frames.iter().zip([1, 0, 1]) {
        assert_eq!(entry.1, audio.frames[index].as_slice());
        assert_eq!(entry.1.as_ptr(), audio.frames[index].as_ptr());
    }
    // This collection must not clamp image positions, alter tokens or turn a
    // repeated source index into one destination: both target rows are live.
    assert_eq!(images[0].2, (usize::MAX, 1));
    assert_eq!(images[2].0, 127);
    assert_eq!(tokens[64].token, 81);
    assert!(tokens[127].logits);
}

#[test]
fn partial_chunks_use_only_live_rows_and_each_inputs_own_media() {
    let first_image = image();
    let mut second_image = image();
    second_image.pixels[0].fill(bf16::from_f32(0.875));
    second_image.positions[0] = (11, 12);
    let first_audio = audio();
    let mut second_audio = audio();
    second_audio.frames[0].fill(bf16::from_f32(0.0625));
    let mut tokens = inputs(P128::SIZE, Some(&first_image), Some(&first_audio));
    tokens[15].soft = Some(SoftToken::Image(0));
    tokens[16].soft = Some(SoftToken::Audio(0));
    tokens[63].soft = Some(SoftToken::Image(0));
    tokens[63].image = Some(&second_image);
    tokens[64].soft = Some(SoftToken::Audio(0));
    tokens[64].audio = Some(&second_audio);
    // The excluded final row is deliberately invalid. Padding must not resolve it.
    tokens[127].soft = Some(SoftToken::Audio(usize::MAX));
    let blocks = runtime::Blocks::default();
    let sequence = Sequence { inputs: &tokens[..127], start_position: 0, blocks: &blocks };
    let (images, frames) = collect_media_rows(&sequence).unwrap();
    assert_eq!(images.iter().map(|r| r.0).collect::<Vec<_>>(), [15, 63]);
    assert_eq!(frames.iter().map(|r| r.0).collect::<Vec<_>>(), [16, 64]);
    assert_eq!(images[0].1.as_ptr(), first_image.pixels[0].as_ptr());
    assert_eq!(images[1].1.as_ptr(), second_image.pixels[0].as_ptr());
    assert_eq!(images[1].2, (11, 12));
    assert_eq!(frames[0].1.as_ptr(), first_audio.frames[0].as_ptr());
    assert_eq!(frames[1].1.as_ptr(), second_audio.frames[0].as_ptr());
}

#[test]
fn plain_text_does_not_resolve_unused_or_invalid_media_payloads() {
    let malformed_image = ImagePatches { pixels: vec![vec![]], positions: vec![] };
    let malformed_audio = AudioFrames { frames: vec![vec![]] };
    let tokens = inputs(P128::SIZE, Some(&malformed_image), Some(&malformed_audio));
    let blocks = runtime::Blocks::default();
    let (images, frames) = collect_media_rows(&Sequence {
        inputs: &tokens, start_position: 128, blocks: &blocks,
    }).unwrap();
    assert!(images.is_empty() && frames.is_empty());
}

#[test]
fn missing_media_and_out_of_range_indices_return_errors() {
    let image = image();
    let audio = audio();
    let blocks = runtime::Blocks::default();
    let mut tokens = inputs(32, None, None);
    for (soft, expected) in [
        (SoftToken::Image(0), "image is missing"),
        (SoftToken::Audio(0), "audio is missing"),
    ] {
        tokens[16].soft = Some(soft);
        let error = collect_media_rows(&Sequence {
            inputs: &tokens, start_position: 0, blocks: &blocks,
        }).err().unwrap();
        assert_eq!(error.to_string(), expected);
    }
    tokens[16].image = Some(&image);
    tokens[16].audio = Some(&audio);
    for (soft, expected) in [
        (SoftToken::Image(2), "image patch index is out of bounds"),
        (SoftToken::Audio(usize::MAX), "audio frame index is out of bounds"),
    ] {
        tokens[16].soft = Some(soft);
        let error = collect_media_rows(&Sequence {
            inputs: &tokens, start_position: 0, blocks: &blocks,
        }).err().unwrap();
        assert_eq!(error.to_string(), expected);
    }
}

#[test]
fn malformed_live_frame_shapes_and_missing_positions_return_errors() {
    let mut malformed_image = image();
    let mut malformed_audio = audio();
    malformed_image.pixels[1].pop();
    malformed_audio.frames[1].push(bf16::from_f32(0.0));
    let missing_position = ImagePatches {
        pixels: vec![vec![bf16::from_f32(0.0); Rv::SIZE]],
        positions: vec![],
    };
    let blocks = runtime::Blocks::default();
    let mut tokens = inputs(32, Some(&malformed_image), Some(&malformed_audio));
    for (soft, expected) in [
        (SoftToken::Image(1), "vision patch must be Rv-wide"),
        (SoftToken::Audio(1), "audio frame must be Aa-wide"),
    ] {
        tokens[15].soft = Some(soft);
        let error = collect_media_rows(&Sequence {
            inputs: &tokens, start_position: 0, blocks: &blocks,
        }).err().unwrap();
        assert_eq!(error.to_string(), expected);
    }
    tokens[15].soft = Some(SoftToken::Image(0));
    tokens[15].image = Some(&missing_position);
    let error = collect_media_rows(&Sequence {
        inputs: &tokens, start_position: 0, blocks: &blocks,
    }).err().unwrap();
    assert_eq!(error.to_string(), "image position index is out of bounds");
}
