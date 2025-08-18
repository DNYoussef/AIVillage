use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::ptr;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::runtime::Runtime;

/// BNContext holds the Tokio runtime used for operations.
#[repr(C)]
pub struct BNContext {
    rt: Runtime,
}

/// BNSession wraps a TCP stream used for echo demo.
#[repr(C)]
pub struct BNSession {
    stream: TcpStream,
    rt_handle: tokio::runtime::Handle,
}

fn to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
}

/// Initialize Betanet context. Currently config path is unused.
#[no_mangle]
pub extern "C" fn bn_init(_cfg_path: *const c_char) -> *mut BNContext {
    let rt = Runtime::new().expect("runtime");
    Box::into_raw(Box::new(BNContext { rt }))
}

/// Dial a connection or listen for one depending on origin string.
/// If origin starts with "listen:", a listener is created and first connection accepted.
#[no_mangle]
pub extern "C" fn bn_dial(ctx: *mut BNContext, origin: *const c_char) -> *mut BNSession {
    if ctx.is_null() {
        return ptr::null_mut();
    }
    let ctx = unsafe { &mut *ctx };
    let origin = to_string(origin);
    let handle = ctx.rt.handle().clone();
    let fut = async move {
        if let Some(rest) = origin.strip_prefix("listen:") {
            let listener = TcpListener::bind(rest).await.ok()?;
            let (stream, _) = listener.accept().await.ok()?;
            Some(stream)
        } else {
            TcpStream::connect(origin).await.ok()
        }
    };
    match ctx.rt.block_on(fut) {
        Some(stream) => Box::into_raw(Box::new(BNSession {
            stream,
            rt_handle: handle,
        })),
        None => ptr::null_mut(),
    }
}

/// Write to stream. Returns number of bytes written or negative error code.
#[no_mangle]
pub extern "C" fn bn_stream_write(sess: *mut BNSession, data: *const u8, len: usize) -> c_int {
    if sess.is_null() {
        return -1;
    }
    if data.is_null() {
        return -2;
    }
    let sess = unsafe { &mut *sess };
    let slice = unsafe { std::slice::from_raw_parts(data, len) };
    let fut = sess.stream.write_all(slice);
    match sess.rt_handle.block_on(fut) {
        Ok(_) => len as c_int,
        Err(_) => -3,
    }
}

/// Read from stream. Returns number of bytes read or negative error code.
#[no_mangle]
pub extern "C" fn bn_stream_read(sess: *mut BNSession, buf: *mut u8, len: usize) -> c_int {
    if sess.is_null() {
        return -1;
    }
    if buf.is_null() {
        return -2;
    }
    let sess = unsafe { &mut *sess };
    let slice = unsafe { std::slice::from_raw_parts_mut(buf, len) };
    let fut = sess.stream.read(slice);
    match sess.rt_handle.block_on(fut) {
        Ok(n) => n as c_int,
        Err(_) => -3,
    }
}

/// Close session and free resources.
#[no_mangle]
pub extern "C" fn bn_close(sess: *mut BNSession) {
    if sess.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(sess));
    }
}

/// Free context.
#[no_mangle]
pub extern "C" fn bn_free(ctx: *mut BNContext) {
    if ctx.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(ctx));
    }
}
