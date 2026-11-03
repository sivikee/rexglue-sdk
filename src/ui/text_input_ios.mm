#import <UIKit/UIKit.h>

#include <rex/ui/window_sdl.h>

// UIAlertController owns the action blocks, which keep this session alive.
// The weak alert reference avoids a cycle. Completion is consumed exactly once.
@interface RexTextInputSession : NSObject <UITextFieldDelegate> {
 @public
  std::function<void(bool, std::string)> completion;
  size_t maxLength;
}
@property(nonatomic, weak) UIAlertController* alert;
- (void)finish:(BOOL)accepted;
- (void)interrupted:(NSNotification*)notification;
@end

@implementation RexTextInputSession
- (void)dealloc {
  [[NSNotificationCenter defaultCenter] removeObserver:self];
}

- (void)finish:(BOOL)accepted {
  if (!completion) {
    return;
  }
  auto callback = std::move(completion);
  completion = nullptr;
  [[NSNotificationCenter defaultCenter] removeObserver:self];
  UIAlertController* alert = self.alert;
  NSString* text = alert.textFields.firstObject.text ?: @"";
  // The guest limit is UTF-16 code units, excluding its terminating NUL.
  // Never cut a surrogate pair or a composed character in half.
  if (text.length > maxLength) {
    NSRange boundary = [text rangeOfComposedCharacterSequenceAtIndex:maxLength];
    text = [text substringToIndex:boundary.location];
  }
  std::string result = accepted ? std::string(text.UTF8String) : std::string();
  [alert.view endEditing:YES];
  if (!alert.presentingViewController) {
    callback(accepted, result);
    return;
  }
  // Dismiss before waking the guest, so gameplay cannot resume under the keyboard.
  [alert dismissViewControllerAnimated:NO completion:^{ callback(accepted, result); }];
}

- (BOOL)textFieldShouldReturn:(UITextField*)textField {
  [self finish:YES];
  return NO;
}

- (void)interrupted:(NSNotification*)notification {
  [self finish:NO];
}
@end

namespace rex::ui {

bool WindowSDL::ShowNativeTextInput(const std::string& title, const std::string& description,
                                    const std::string& default_text, size_t max_length,
                                    std::function<void(bool, std::string)> completion) {
  if (!sdl_window_) {
    return false;
  }
  UIWindow* window = (__bridge UIWindow*)SDL_GetPointerProperty(
      SDL_GetWindowProperties(sdl_window_), SDL_PROP_WINDOW_UIKIT_WINDOW_POINTER, nullptr);
  UIViewController* presenter = window.rootViewController;
  if (!presenter.view.window || presenter.presentedViewController ||
      UIApplication.sharedApplication.applicationState != UIApplicationStateActive) {
    return false;
  }

  // SDL observes every UIKit keyboard, even when its hidden text field is not
  // focused. Clear its input rectangle so it cannot pan the game underneath
  // this alert in response to keyboard notifications.
  SDL_SetTextInputArea(sdl_window_, nullptr, 0);
  SetTextInputActive(false);
  SDL_StopTextInput(sdl_window_);

  NSString* nativeTitle = [NSString stringWithUTF8String:title.c_str()];
  if (!nativeTitle.length) {
    nativeTitle = @"Keyboard Input";
  }
  UIAlertController* alert = [UIAlertController
      alertControllerWithTitle:nativeTitle
                       message:[NSString stringWithUTF8String:description.c_str()]
                preferredStyle:UIAlertControllerStyleAlert];
  RexTextInputSession* session = [RexTextInputSession new];
  session->completion = std::move(completion);
  session->maxLength = max_length;
  session.alert = alert;
  [alert addTextFieldWithConfigurationHandler:^(UITextField* field) {
    field.text = [NSString stringWithUTF8String:default_text.c_str()];
    field.autocorrectionType = UITextAutocorrectionTypeNo;
    field.spellCheckingType = UITextSpellCheckingTypeNo;
    field.returnKeyType = UIReturnKeyDone;
    field.delegate = session;
  }];
  [alert addAction:[UIAlertAction actionWithTitle:@"Cancel" style:UIAlertActionStyleCancel
                                        handler:^(UIAlertAction*) { [session finish:NO]; }]];
  UIAlertAction* accept = [UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault
                                               handler:^(UIAlertAction*) { [session finish:YES]; }];
  [alert addAction:accept];
  alert.preferredAction = accept;
  [[NSNotificationCenter defaultCenter] addObserver:session selector:@selector(interrupted:)
                                               name:UIApplicationWillResignActiveNotification
                                             object:nil];
  [presenter presentViewController:alert animated:YES completion:nil];
  return true;
}

}  // namespace rex::ui
