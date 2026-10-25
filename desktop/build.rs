// Embeds the NoMusic icon (assets/nomusic.ico, made from the extension's icons) into the executables.
fn main() {
    println!("cargo:rerun-if-changed=assets/nomusic.rc");
    println!("cargo:rerun-if-changed=assets/nomusic.ico");
    embed_resource::compile("assets/nomusic.rc", embed_resource::NONE).manifest_optional().unwrap();
}
