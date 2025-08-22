use utlsgen::{compute_ja3, compute_ja4, template_chrome_n, template_chrome_n_minus_2};

#[test]
fn test_chrome_n_fingerprint() {
    let t = template_chrome_n();
    assert_eq!(compute_ja3(&t.metadata), t.expected_ja3);
    assert_eq!(compute_ja4(&t.metadata), t.expected_ja4);
}

#[test]
fn test_chrome_n_minus_2_fingerprint() {
    let t = template_chrome_n_minus_2();
    assert_eq!(compute_ja3(&t.metadata), t.expected_ja3);
    assert_eq!(compute_ja4(&t.metadata), t.expected_ja4);
}
