"""
Data generation and handling module for LazyMode model.

This module provides functions to generate synthetic training data
and load datasets for training the LazyMode model.
"""

from typing import Dict, List, Tuple
import json
import os


# Template patterns for generating training data
ISSUE_TEMPLATES = [
    {
        "input": "App crashes on login button tap",
        "output": """## Bug Report: App Crashes on Login Button Tap

### Description
The application crashes when users tap the login button on the authentication screen.

### Environment
- **Platform**: Mobile Application
- **Component**: Authentication/Login

### Steps to Reproduce
1. Open the application
2. Navigate to the login screen
3. Enter credentials (any valid or invalid)
4. Tap the login button
5. Observe the crash

### Expected Behavior
The application should process the login attempt and either authenticate the user or display an error message.

### Actual Behavior
The application crashes immediately upon tapping the login button.

### Error Logs
```
// Add crash logs here
```

### Proposed Tasks
- [ ] Investigate crash logs to identify root cause
- [ ] Add null checks for login button handler
- [ ] Implement proper error handling
- [ ] Add unit tests for login functionality
- [ ] Test fix on all supported platforms"""
    },
    {
        "input": "Database connection times out after 30 seconds",
        "output": """## Bug Report: Database Connection Timeout

### Description
The database connection is timing out after 30 seconds, causing service disruption.

### Environment
- **Platform**: Backend Service
- **Component**: Database Layer

### Steps to Reproduce
1. Start the application server
2. Initiate a database query
3. Wait for 30 seconds
4. Observe connection timeout error

### Expected Behavior
Database connections should be established within a reasonable timeframe (< 5 seconds) or properly handle long-running queries.

### Actual Behavior
All database connections timeout after exactly 30 seconds, indicating a configuration issue.

### Error Logs
```
Connection timeout after 30000ms
```

### Proposed Tasks
- [ ] Review database connection pool configuration
- [ ] Check network connectivity between app and database
- [ ] Increase timeout value or implement retry logic
- [ ] Add connection health checks
- [ ] Monitor database server performance"""
    },
    {
        "input": "User profile picture not loading on homepage",
        "output": """## Bug Report: User Profile Picture Not Loading

### Description
User profile pictures are not loading on the homepage, showing a placeholder or broken image instead.

### Environment
- **Platform**: Web Application
- **Component**: User Interface / Media

### Steps to Reproduce
1. Log into the application
2. Navigate to the homepage
3. Look at the user profile section
4. Observe that the profile picture is not displayed

### Expected Behavior
The user's profile picture should load and display correctly on the homepage.

### Actual Behavior
Profile pictures show as broken images or placeholders.

### Error Logs
```
404 Not Found: /api/users/{id}/avatar
```

### Proposed Tasks
- [ ] Verify image storage configuration
- [ ] Check image CDN connectivity
- [ ] Review CORS settings for image requests
- [ ] Implement fallback placeholder image
- [ ] Add image loading error handling"""
    },
    {
        "input": "Search feature returns no results for valid queries",
        "output": """## Bug Report: Search Returns No Results

### Description
The search feature returns no results even when searching for items that definitely exist in the database.

### Environment
- **Platform**: Web/Mobile Application
- **Component**: Search Functionality

### Steps to Reproduce
1. Navigate to the search page
2. Enter a known valid search term
3. Submit the search
4. Observe empty results

### Expected Behavior
Search should return relevant results matching the query.

### Actual Behavior
Search returns zero results for all queries.

### Error Logs
```
Search index not initialized or empty
```

### Proposed Tasks
- [ ] Verify search index is properly built
- [ ] Check indexing service status
- [ ] Review search query parsing logic
- [ ] Implement search result caching
- [ ] Add search analytics for debugging"""
    },
    {
        "input": "Payment processing fails with credit card",
        "output": """## Bug Report: Payment Processing Failure

### Description
Credit card payments are failing during checkout, preventing users from completing purchases.

### Environment
- **Platform**: E-commerce Platform
- **Component**: Payment Gateway

### Steps to Reproduce
1. Add items to cart
2. Proceed to checkout
3. Enter valid credit card information
4. Submit payment
5. Observe payment failure

### Expected Behavior
Valid credit card transactions should process successfully.

### Actual Behavior
All credit card payments fail with a generic error message.

### Error Logs
```
Payment gateway error: Invalid merchant credentials
```

### Proposed Tasks
- [ ] Verify payment gateway API credentials
- [ ] Check SSL certificate validity
- [ ] Review payment gateway integration code
- [ ] Test with payment gateway sandbox
- [ ] Implement detailed error logging"""
    },
    {
        "input": "Email notifications not being sent",
        "output": """## Bug Report: Email Notifications Not Sending

### Description
Users are not receiving email notifications for important events like password resets and order confirmations.

### Environment
- **Platform**: Backend Service
- **Component**: Email Service

### Steps to Reproduce
1. Trigger an event that should send an email (e.g., password reset)
2. Check the recipient's inbox
3. Wait several minutes
4. Check spam folder
5. Observe no email received

### Expected Behavior
Email notifications should be sent and delivered to users within minutes.

### Actual Behavior
No emails are being sent or delivered.

### Error Logs
```
SMTP connection refused: Host smtp.example.com:587
```

### Proposed Tasks
- [ ] Verify SMTP server configuration
- [ ] Check email service credentials
- [ ] Review email queue processing
- [ ] Test email delivery with different providers
- [ ] Implement email delivery monitoring"""
    },
    {
        "input": "File upload fails for files larger than 5MB",
        "output": """## Bug Report: Large File Upload Failure

### Description
File uploads fail when attempting to upload files larger than 5MB, preventing users from uploading important documents.

### Environment
- **Platform**: Web Application
- **Component**: File Upload Service

### Steps to Reproduce
1. Navigate to file upload section
2. Select a file larger than 5MB
3. Initiate upload
4. Observe upload failure

### Expected Behavior
Files up to the documented size limit should upload successfully.

### Actual Behavior
All files over 5MB fail to upload with a server error.

### Error Logs
```
413 Payload Too Large
```

### Proposed Tasks
- [ ] Review server upload size limits
- [ ] Update nginx/apache configuration
- [ ] Implement chunked file upload
- [ ] Add file size validation on frontend
- [ ] Improve error messaging for users"""
    },
    {
        "input": "Dark mode colors are incorrect in settings page",
        "output": """## Bug Report: Dark Mode Colors Incorrect

### Description
When dark mode is enabled, the settings page displays incorrect colors making text unreadable.

### Environment
- **Platform**: Web/Mobile Application
- **Component**: UI Theme System

### Steps to Reproduce
1. Enable dark mode in settings
2. Navigate to the settings page
3. Observe color inconsistencies

### Expected Behavior
All text and UI elements should be properly themed for dark mode with good contrast.

### Actual Behavior
Some text appears in dark colors on dark backgrounds, making it unreadable.

### Error Logs
```
// No JavaScript errors, CSS issue
```

### Proposed Tasks
- [ ] Audit CSS variables for dark mode
- [ ] Fix color contrast issues
- [ ] Add dark mode styles for settings page
- [ ] Implement automated contrast checking
- [ ] Test across all application pages"""
    },
    {
        "input": "API rate limiting not working correctly",
        "output": """## Bug Report: API Rate Limiting Malfunction

### Description
API rate limiting is either too aggressive or not being applied correctly, affecting legitimate users.

### Environment
- **Platform**: API Service
- **Component**: Rate Limiter

### Steps to Reproduce
1. Make API requests at normal rate
2. Observe rate limit errors before reaching documented limit
3. Or observe no rate limiting when making excessive requests

### Expected Behavior
Rate limiting should accurately track and limit requests per the documented policy.

### Actual Behavior
Rate limits are either triggered prematurely or not at all.

### Error Logs
```
Rate limit counter inconsistency detected
```

### Proposed Tasks
- [ ] Review rate limiting algorithm
- [ ] Check Redis/cache connection for rate tracking
- [ ] Fix counter increment logic
- [ ] Add rate limit status headers
- [ ] Implement rate limit monitoring dashboard"""
    },
    {
        "input": "Mobile app battery drain issue",
        "output": """## Bug Report: Excessive Battery Drain

### Description
The mobile application is causing excessive battery drain, significantly impacting device battery life.

### Environment
- **Platform**: Mobile Application (iOS/Android)
- **Component**: Background Services

### Steps to Reproduce
1. Install the application
2. Use the app normally for 1 hour
3. Check battery usage in device settings
4. Observe unusually high battery consumption

### Expected Behavior
The application should use reasonable battery resources comparable to similar apps.

### Actual Behavior
Battery consumption is 3-5x higher than expected, even when app is in background.

### Error Logs
```
// Battery usage statistics from device
```

### Proposed Tasks
- [ ] Profile app for background activity
- [ ] Review location service usage
- [ ] Optimize network polling intervals
- [ ] Implement proper background task management
- [ ] Add battery usage analytics"""
    },
    {
        "input": "Login session expires too quickly",
        "output": """## Bug Report: Premature Session Expiration

### Description
User login sessions are expiring too quickly, forcing users to log in repeatedly during normal use.

### Environment
- **Platform**: Web Application
- **Component**: Authentication/Session Management

### Steps to Reproduce
1. Log into the application
2. Use the application normally
3. Session expires after short period (< 15 minutes)
4. User is forced to log in again

### Expected Behavior
Sessions should remain active for a reasonable period during active use.

### Actual Behavior
Sessions expire prematurely, disrupting user workflow.

### Error Logs
```
Session token expired before expected time
```

### Proposed Tasks
- [ ] Review session timeout configuration
- [ ] Implement sliding session expiration
- [ ] Add session refresh on user activity
- [ ] Check token expiration logic
- [ ] Add session management monitoring"""
    },
    {
        "input": "Push notifications not working on iOS",
        "output": """## Bug Report: iOS Push Notifications Failing

### Description
Push notifications are not being delivered to iOS devices, while Android devices receive them correctly.

### Environment
- **Platform**: iOS Mobile Application
- **Component**: Push Notification Service

### Steps to Reproduce
1. Install app on iOS device
2. Grant notification permissions
3. Trigger a push notification event
4. Observe that notification is not received

### Expected Behavior
Push notifications should be delivered to iOS devices.

### Actual Behavior
No push notifications are received on iOS devices.

### Error Logs
```
APNS connection failed: Invalid certificate
```

### Proposed Tasks
- [ ] Verify APNS certificate validity
- [ ] Check iOS bundle identifier configuration
- [ ] Review push notification payload format
- [ ] Test with APNS sandbox environment
- [ ] Implement push delivery logging"""
    },
    {
        "input": "Data export generates corrupted CSV file",
        "output": """## Bug Report: Corrupted CSV Export

### Description
When exporting data to CSV format, the generated file is corrupted or improperly formatted.

### Environment
- **Platform**: Web Application
- **Component**: Data Export Service

### Steps to Reproduce
1. Navigate to data export section
2. Select CSV as export format
3. Download the generated file
4. Attempt to open in Excel or CSV viewer
5. Observe formatting errors or data corruption

### Expected Behavior
CSV export should generate properly formatted, valid CSV files.

### Actual Behavior
Generated CSV files have encoding issues, missing columns, or corrupt data.

### Error Logs
```
Character encoding mismatch in export stream
```

### Proposed Tasks
- [ ] Fix character encoding to UTF-8
- [ ] Properly escape special characters
- [ ] Add CSV header row validation
- [ ] Implement export file verification
- [ ] Add download progress indicator"""
    },
    {
        "input": "Page load time is very slow more than 10 seconds",
        "output": """## Bug Report: Slow Page Load Performance

### Description
Page load times exceed 10 seconds, severely impacting user experience.

### Environment
- **Platform**: Web Application
- **Component**: Frontend/Backend Performance

### Steps to Reproduce
1. Navigate to the application
2. Observe page loading
3. Measure time until fully interactive
4. Note load times exceeding 10 seconds

### Expected Behavior
Pages should load within 2-3 seconds for optimal user experience.

### Actual Behavior
Page load times regularly exceed 10 seconds.

### Error Logs
```
// Performance metrics from browser dev tools
```

### Proposed Tasks
- [ ] Profile frontend bundle size
- [ ] Optimize API response times
- [ ] Implement lazy loading for components
- [ ] Add caching strategies
- [ ] Set up performance monitoring"""
    },
    {
        "input": "Two-factor authentication SMS codes not received",
        "output": """## Bug Report: 2FA SMS Codes Not Delivered

### Description
Users are not receiving SMS codes for two-factor authentication, preventing account access.

### Environment
- **Platform**: Authentication Service
- **Component**: SMS/2FA Service

### Steps to Reproduce
1. Enable 2FA on account
2. Log in to trigger 2FA prompt
3. Request SMS code
4. Wait for SMS
5. No SMS is received

### Expected Behavior
SMS verification codes should be delivered within 30 seconds.

### Actual Behavior
SMS codes are never received, locking users out of accounts.

### Error Logs
```
SMS gateway error: Insufficient credits or invalid API key
```

### Proposed Tasks
- [ ] Verify SMS gateway credentials and balance
- [ ] Check phone number formatting
- [ ] Review SMS gateway response codes
- [ ] Implement SMS delivery confirmation
- [ ] Add backup 2FA methods (TOTP)"""
    },
]

# Additional templates to reach ~50 examples
ADDITIONAL_TEMPLATES = [
    {
        "input": "Logout button not working",
        "output": """## Bug Report: Logout Button Not Functioning

### Description
The logout button does not work, leaving users unable to sign out of their accounts.

### Environment
- **Platform**: Web Application
- **Component**: Authentication

### Steps to Reproduce
1. Log into the application
2. Click the logout button
3. Observe nothing happens

### Expected Behavior
Clicking logout should end the session and redirect to login page.

### Actual Behavior
Logout button click has no effect; session remains active.

### Error Logs
```
// Check browser console for JavaScript errors
```

### Proposed Tasks
- [ ] Debug logout click handler
- [ ] Verify session invalidation endpoint
- [ ] Add logout confirmation
- [ ] Test cross-browser compatibility
- [ ] Implement proper session cleanup"""
    },
    {
        "input": "Dropdown menu items are not clickable",
        "output": """## Bug Report: Dropdown Menu Items Unresponsive

### Description
Dropdown menu items cannot be clicked or selected, breaking navigation functionality.

### Environment
- **Platform**: Web Application
- **Component**: Navigation/UI

### Steps to Reproduce
1. Navigate to page with dropdown menu
2. Click to open dropdown
3. Attempt to click menu item
4. Observe clicks do not register

### Expected Behavior
Dropdown menu items should be clickable and navigate to respective sections.

### Actual Behavior
Menu items are visible but unresponsive to clicks.

### Error Logs
```
// Check for z-index or event handler issues
```

### Proposed Tasks
- [ ] Check CSS z-index stacking
- [ ] Verify event handlers are attached
- [ ] Fix pointer-events CSS property
- [ ] Test touch events on mobile
- [ ] Add keyboard navigation support"""
    },
    {
        "input": "Password reset link expired immediately",
        "output": """## Bug Report: Password Reset Link Immediate Expiration

### Description
Password reset links expire immediately or within seconds of being sent, preventing password recovery.

### Environment
- **Platform**: Web Application
- **Component**: Password Recovery

### Steps to Reproduce
1. Request password reset
2. Receive email with reset link
3. Click link immediately
4. See "Link expired" message

### Expected Behavior
Reset links should remain valid for at least 1 hour.

### Actual Behavior
Links expire instantly or within seconds.

### Error Logs
```
Token timestamp validation failed
```

### Proposed Tasks
- [ ] Check server time synchronization
- [ ] Review token expiration logic
- [ ] Extend token validity period
- [ ] Add token debugging logs
- [ ] Implement new token request option"""
    },
    {
        "input": "Calendar events showing wrong timezone",
        "output": """## Bug Report: Calendar Timezone Display Issue

### Description
Calendar events are displayed in the wrong timezone, causing scheduling confusion.

### Environment
- **Platform**: Web/Mobile Application
- **Component**: Calendar/Scheduling

### Steps to Reproduce
1. Create calendar event in local timezone
2. View event on calendar
3. Observe time is displayed in different timezone

### Expected Behavior
Events should display in user's local timezone or clearly indicate the timezone.

### Actual Behavior
Events show incorrect times due to timezone mismatch.

### Error Logs
```
Timezone conversion error: UTC offset mismatch
```

### Proposed Tasks
- [ ] Store and display user timezone preference
- [ ] Fix timezone conversion logic
- [ ] Add timezone indicator to events
- [ ] Handle daylight saving time correctly
- [ ] Test across multiple timezones"""
    },
    {
        "input": "Image gallery swipe not smooth on mobile",
        "output": """## Bug Report: Image Gallery Swipe Performance

### Description
Image gallery swipe gestures are laggy and unresponsive on mobile devices.

### Environment
- **Platform**: Mobile Web/App
- **Component**: Image Gallery

### Steps to Reproduce
1. Open image gallery on mobile device
2. Attempt to swipe between images
3. Observe stuttering and lag

### Expected Behavior
Image transitions should be smooth and responsive (60fps).

### Actual Behavior
Swipe gestures are choppy with noticeable lag.

### Error Logs
```
// Frame drops in performance monitor
```

### Proposed Tasks
- [ ] Optimize image loading and caching
- [ ] Implement hardware-accelerated animations
- [ ] Reduce image resolution for thumbnails
- [ ] Add lazy loading for off-screen images
- [ ] Profile and fix memory leaks"""
    },
    {
        "input": "Form validation errors not showing",
        "output": """## Bug Report: Form Validation Errors Hidden

### Description
Form validation errors are not displayed to users, leaving them confused about input requirements.

### Environment
- **Platform**: Web Application
- **Component**: Form Handling

### Steps to Reproduce
1. Navigate to form page
2. Submit form with invalid data
3. Form submission fails silently
4. No error messages displayed

### Expected Behavior
Clear validation error messages should appear next to invalid fields.

### Actual Behavior
No visual feedback for validation errors.

### Error Logs
```
Validation errors generated but not rendered
```

### Proposed Tasks
- [ ] Connect validation state to UI
- [ ] Style error message components
- [ ] Add field-level error indicators
- [ ] Implement form-level error summary
- [ ] Add accessibility for error messages"""
    },
    {
        "input": "Audio player stops when phone screen locks",
        "output": """## Bug Report: Audio Stops on Screen Lock

### Description
Audio playback stops when the phone screen locks, interrupting media consumption.

### Environment
- **Platform**: Mobile Application
- **Component**: Media Player

### Steps to Reproduce
1. Start audio playback
2. Lock phone screen
3. Audio stops playing

### Expected Behavior
Audio should continue playing in background when screen is locked.

### Actual Behavior
Audio playback stops immediately when screen locks.

### Error Logs
```
Background audio session not configured
```

### Proposed Tasks
- [ ] Configure background audio mode
- [ ] Request background execution permissions
- [ ] Handle audio session interruptions
- [ ] Add lock screen controls
- [ ] Test background playback thoroughly"""
    },
    {
        "input": "Print preview shows blank pages",
        "output": """## Bug Report: Print Preview Shows Blank Pages

### Description
The print preview displays blank pages instead of the document content.

### Environment
- **Platform**: Web Application
- **Component**: Print Functionality

### Steps to Reproduce
1. Navigate to document view
2. Select print option
3. View print preview
4. Observe blank pages

### Expected Behavior
Print preview should accurately display document content.

### Actual Behavior
Print preview shows completely blank pages.

### Error Logs
```
// Check print media CSS and content rendering
```

### Proposed Tasks
- [ ] Add @media print CSS rules
- [ ] Fix content visibility in print mode
- [ ] Handle page breaks correctly
- [ ] Remove print-hidden elements
- [ ] Test across browsers"""
    },
    {
        "input": "Autocomplete suggestions not appearing",
        "output": """## Bug Report: Autocomplete Not Showing Suggestions

### Description
Autocomplete/typeahead suggestions are not appearing in search or input fields.

### Environment
- **Platform**: Web Application
- **Component**: Search/Input

### Steps to Reproduce
1. Click on autocomplete-enabled input field
2. Start typing
3. Wait for suggestions
4. No suggestions appear

### Expected Behavior
Relevant suggestions should appear as user types.

### Actual Behavior
No autocomplete suggestions are displayed.

### Error Logs
```
Autocomplete API request failed or returned empty
```

### Proposed Tasks
- [ ] Verify autocomplete API endpoint
- [ ] Check minimum character trigger
- [ ] Debug suggestion data fetch
- [ ] Fix suggestion dropdown rendering
- [ ] Add loading indicator"""
    },
    {
        "input": "Infinite scroll loads same items repeatedly",
        "output": """## Bug Report: Infinite Scroll Duplicate Loading

### Description
Infinite scroll feature loads the same items repeatedly instead of loading new content.

### Environment
- **Platform**: Web Application
- **Component**: Pagination/Loading

### Steps to Reproduce
1. Navigate to list with infinite scroll
2. Scroll to trigger loading
3. Observe same items loading again
4. Continue scrolling, duplicates persist

### Expected Behavior
Each scroll should load new, unique items.

### Actual Behavior
The same items are loaded repeatedly, creating duplicates.

### Error Logs
```
Pagination offset not incrementing
```

### Proposed Tasks
- [ ] Fix pagination cursor/offset tracking
- [ ] Deduplicate loaded items
- [ ] Add end-of-list detection
- [ ] Implement proper scroll position tracking
- [ ] Add loading state indicator"""
    },
    {
        "input": "Drag and drop not working on Firefox",
        "output": """## Bug Report: Drag and Drop Firefox Incompatibility

### Description
Drag and drop functionality does not work on Firefox browser.

### Environment
- **Platform**: Web Application (Firefox)
- **Component**: Drag and Drop

### Steps to Reproduce
1. Open application in Firefox
2. Attempt to drag an element
3. Observe drag operation fails

### Expected Behavior
Drag and drop should work consistently across all major browsers.

### Actual Behavior
Drag and drop only works in Chrome, fails in Firefox.

### Error Logs
```
// Firefox console errors for drag events
```

### Proposed Tasks
- [ ] Review drag event implementation
- [ ] Add Firefox-specific event handling
- [ ] Use cross-browser drag library
- [ ] Test on all major browsers
- [ ] Add browser capability detection"""
    },
    {
        "input": "Language selection not persisting",
        "output": """## Bug Report: Language Preference Not Saved

### Description
Selected language preference resets on every page load or session.

### Environment
- **Platform**: Web Application
- **Component**: Localization

### Steps to Reproduce
1. Change language from default
2. Navigate to another page or refresh
3. Observe language reverted to default

### Expected Behavior
Language preference should persist across sessions.

### Actual Behavior
Language resets to default on each visit.

### Error Logs
```
Language preference not stored in cookies/storage
```

### Proposed Tasks
- [ ] Store language preference in localStorage
- [ ] Add language cookie for server-side rendering
- [ ] Sync preference to user profile if logged in
- [ ] Handle language header from browser
- [ ] Test preference persistence"""
    },
    {
        "input": "Progress bar not updating during file upload",
        "output": """## Bug Report: Upload Progress Bar Static

### Description
The file upload progress bar does not update, staying at 0% throughout the upload.

### Environment
- **Platform**: Web Application
- **Component**: File Upload UI

### Steps to Reproduce
1. Select file to upload
2. Initiate upload
3. Observe progress bar stays at 0%
4. Upload completes but bar never updated

### Expected Behavior
Progress bar should reflect actual upload progress.

### Actual Behavior
Progress bar remains static during entire upload.

### Error Logs
```
Progress event handler not attached
```

### Proposed Tasks
- [ ] Attach progress event listener to XHR/fetch
- [ ] Calculate and display percentage
- [ ] Update progress bar UI
- [ ] Add upload speed indicator
- [ ] Implement cancel upload option"""
    },
    {
        "input": "Copy to clipboard not working on Safari",
        "output": """## Bug Report: Copy to Clipboard Safari Issue

### Description
Copy to clipboard functionality does not work on Safari browser.

### Environment
- **Platform**: Web Application (Safari)
- **Component**: Clipboard API

### Steps to Reproduce
1. Open application in Safari
2. Click copy button
3. Attempt to paste
4. Nothing was copied

### Expected Behavior
Content should be copied to clipboard in all browsers.

### Actual Behavior
Copy works in Chrome/Firefox but fails in Safari.

### Error Logs
```
Clipboard API not permitted in this context
```

### Proposed Tasks
- [ ] Use legacy document.execCommand fallback
- [ ] Request clipboard permissions properly
- [ ] Add Safari-specific clipboard handling
- [ ] Test on iOS Safari as well
- [ ] Provide feedback on copy success/failure"""
    },
    {
        "input": "Modal dialog closes when clicking inside",
        "output": """## Bug Report: Modal Closes Unexpectedly

### Description
Modal dialogs close when clicking anywhere inside them, not just the close button.

### Environment
- **Platform**: Web Application
- **Component**: Modal/Dialog UI

### Steps to Reproduce
1. Open a modal dialog
2. Click anywhere inside the modal content
3. Modal closes unexpectedly

### Expected Behavior
Modal should only close when clicking close button or overlay backdrop.

### Actual Behavior
Any click inside modal causes it to close.

### Error Logs
```
// Click event propagation issue
```

### Proposed Tasks
- [ ] Stop event propagation on modal content
- [ ] Separate backdrop and content click handlers
- [ ] Add proper close button handling
- [ ] Implement escape key to close
- [ ] Test modal interaction patterns"""
    },
    {
        "input": "Video thumbnail not generating for uploads",
        "output": """## Bug Report: Video Thumbnails Not Generated

### Description
Uploaded videos do not have automatically generated thumbnails.

### Environment
- **Platform**: Web Application
- **Component**: Video Processing

### Steps to Reproduce
1. Upload a video file
2. Wait for processing to complete
3. View video in gallery
4. Observe missing thumbnail

### Expected Behavior
System should auto-generate thumbnail from video frame.

### Actual Behavior
Videos display with placeholder instead of thumbnail.

### Error Logs
```
FFmpeg thumbnail extraction failed
```

### Proposed Tasks
- [ ] Verify FFmpeg installation and path
- [ ] Check video format compatibility
- [ ] Handle thumbnail generation errors
- [ ] Implement manual thumbnail upload option
- [ ] Add thumbnail generation queue"""
    },
    {
        "input": "Chart data labels overlapping",
        "output": """## Bug Report: Chart Data Labels Overlap

### Description
Data labels on charts overlap with each other, making them unreadable.

### Environment
- **Platform**: Web Application
- **Component**: Data Visualization

### Steps to Reproduce
1. Navigate to chart/dashboard page
2. View chart with multiple data points
3. Observe overlapping labels

### Expected Behavior
Labels should be positioned without overlapping.

### Actual Behavior
Multiple labels overlap, obscuring the data.

### Error Logs
```
// Chart rendering issue, no console errors
```

### Proposed Tasks
- [ ] Implement label collision detection
- [ ] Add smart label positioning
- [ ] Use label rotation for dense data
- [ ] Implement hover-to-show labels
- [ ] Add zoom functionality for detail view"""
    },
    {
        "input": "Touch ID authentication not prompting",
        "output": """## Bug Report: Touch ID/Biometric Not Prompting

### Description
The app does not prompt for Touch ID or biometric authentication when expected.

### Environment
- **Platform**: Mobile Application
- **Component**: Biometric Authentication

### Steps to Reproduce
1. Enable biometric login in settings
2. Close and reopen app
3. Expect biometric prompt
4. Only password prompt appears

### Expected Behavior
App should prompt for biometric authentication when configured.

### Actual Behavior
Biometric prompt never appears despite being enabled.

### Error Logs
```
Biometric authentication not available or not enrolled
```

### Proposed Tasks
- [ ] Check biometric availability on device
- [ ] Verify biometric preference is saved
- [ ] Handle biometric API errors
- [ ] Fall back gracefully to password
- [ ] Add biometric enrollment guidance"""
    },
    {
        "input": "Webhook notifications delayed by hours",
        "output": """## Bug Report: Webhook Delivery Delays

### Description
Webhook notifications are being delivered hours after the triggering event occurs.

### Environment
- **Platform**: API/Backend Service
- **Component**: Webhook Service

### Steps to Reproduce
1. Configure webhook endpoint
2. Trigger event that sends webhook
3. Monitor webhook endpoint
4. Observe webhook arrives hours later

### Expected Behavior
Webhooks should be delivered within seconds of the triggering event.

### Actual Behavior
Webhooks are delayed by hours, sometimes arriving the next day.

### Error Logs
```
Webhook queue backlog: 50000+ pending
```

### Proposed Tasks
- [ ] Scale webhook delivery workers
- [ ] Implement priority queue for recent events
- [ ] Add retry mechanism with backoff
- [ ] Monitor queue depth and latency
- [ ] Alert on delivery delays"""
    },
    {
        "input": "SSO login redirects to error page",
        "output": """## Bug Report: SSO Login Redirect Error

### Description
Single Sign-On (SSO) login attempts redirect to an error page instead of completing authentication.

### Environment
- **Platform**: Web Application
- **Component**: SSO/OAuth

### Steps to Reproduce
1. Click SSO login button
2. Authenticate with identity provider
3. Get redirected back to application
4. See error page instead of dashboard

### Expected Behavior
SSO authentication should complete and redirect to dashboard.

### Actual Behavior
Redirect returns to error page after identity provider authentication.

### Error Logs
```
OAuth callback validation failed: state mismatch
```

### Proposed Tasks
- [ ] Verify OAuth callback URL configuration
- [ ] Check state parameter handling
- [ ] Review identity provider settings
- [ ] Add detailed error logging
- [ ] Implement SSO debugging mode"""
    },
    {
        "input": "Accessibility screen reader not reading content",
        "output": """## Bug Report: Screen Reader Accessibility Issue

### Description
Screen readers are not properly reading page content, affecting accessibility for visually impaired users.

### Environment
- **Platform**: Web Application
- **Component**: Accessibility

### Steps to Reproduce
1. Enable screen reader (VoiceOver, NVDA, etc.)
2. Navigate to application
3. Observe content is skipped or misread

### Expected Behavior
All content should be properly announced by screen readers.

### Actual Behavior
Important content is skipped or reading order is incorrect.

### Error Logs
```
Missing ARIA labels and roles
```

### Proposed Tasks
- [ ] Add proper ARIA labels and roles
- [ ] Fix heading hierarchy
- [ ] Ensure logical tab order
- [ ] Add alt text to images
- [ ] Conduct accessibility audit"""
    },
    {
        "input": "QR code scanner not focusing camera",
        "output": """## Bug Report: QR Scanner Camera Focus Issue

### Description
The QR code scanner camera does not auto-focus, making it difficult to scan codes.

### Environment
- **Platform**: Mobile Application
- **Component**: Camera/QR Scanner

### Steps to Reproduce
1. Open QR scanner feature
2. Point camera at QR code
3. Observe blurry/unfocused camera view
4. Unable to scan code

### Expected Behavior
Camera should auto-focus on QR codes for quick scanning.

### Actual Behavior
Camera remains out of focus, codes cannot be scanned.

### Error Logs
```
Camera autofocus mode not configured
```

### Proposed Tasks
- [ ] Enable continuous autofocus mode
- [ ] Add tap-to-focus functionality
- [ ] Optimize camera settings for scanning
- [ ] Handle low-light conditions
- [ ] Add manual focus controls"""
    },
    {
        "input": "Exported PDF has missing fonts",
        "output": """## Bug Report: PDF Export Missing Fonts

### Description
Exported PDF documents have missing fonts, causing text to display incorrectly.

### Environment
- **Platform**: Web Application
- **Component**: PDF Generation

### Steps to Reproduce
1. Create document with custom fonts
2. Export to PDF
3. Open PDF in viewer
4. Observe fonts are substituted or missing

### Expected Behavior
PDF should embed fonts or use compatible alternatives.

### Actual Behavior
Custom fonts are not embedded, causing display issues.

### Error Logs
```
Font embedding failed: license restriction
```

### Proposed Tasks
- [ ] Embed fonts in PDF or convert to paths
- [ ] Use web-safe font fallbacks
- [ ] Verify font licensing for embedding
- [ ] Test PDF across different viewers
- [ ] Add font embedding options"""
    },
    {
        "input": "Bulk delete only deletes first item",
        "output": """## Bug Report: Bulk Delete Only Removes One Item

### Description
The bulk delete feature only deletes the first selected item instead of all selected items.

### Environment
- **Platform**: Web Application
- **Component**: Bulk Operations

### Steps to Reproduce
1. Select multiple items
2. Click bulk delete
3. Confirm deletion
4. Only first item is deleted

### Expected Behavior
All selected items should be deleted.

### Actual Behavior
Only the first item in selection is deleted.

### Error Logs
```
Bulk delete only processing first ID in array
```

### Proposed Tasks
- [ ] Fix iteration over selected items
- [ ] Pass complete ID array to delete API
- [ ] Handle partial deletion failures
- [ ] Add progress indicator for bulk operations
- [ ] Implement undo for bulk delete"""
    },
    {
        "input": "Real-time notifications not updating",
        "output": """## Bug Report: Real-time Notifications Not Updating

### Description
Real-time notifications are not being received or displayed, requiring page refresh.

### Environment
- **Platform**: Web Application
- **Component**: WebSocket/Notifications

### Steps to Reproduce
1. Open application
2. Trigger notification-worthy event
3. Observe notification does not appear
4. Refresh page to see notification

### Expected Behavior
Notifications should appear in real-time without refresh.

### Actual Behavior
Notifications only visible after manual page refresh.

### Error Logs
```
WebSocket connection closed unexpectedly
```

### Proposed Tasks
- [ ] Debug WebSocket connection stability
- [ ] Implement automatic reconnection
- [ ] Add connection status indicator
- [ ] Fall back to polling if WebSocket fails
- [ ] Test notification delivery reliability"""
    },
]

# Additional templates to reach 50
MORE_TEMPLATES = [
    {
        "input": "Sorting does not work correctly for dates",
        "output": """## Bug Report: Date Sorting Incorrect

### Description
Date columns are not sorting correctly, treating dates as strings instead of date values.

### Environment
- **Platform**: Web Application
- **Component**: Data Table/Sorting

### Steps to Reproduce
1. Navigate to table with date column
2. Click to sort by date
3. Observe incorrect ordering

### Expected Behavior
Dates should sort chronologically.

### Actual Behavior
Dates sort alphabetically (e.g., "2" comes after "1" regardless of month).

### Error Logs
```
// No errors, logic issue in comparator
```

### Proposed Tasks
- [ ] Parse dates before comparison
- [ ] Use date library for sorting
- [ ] Handle different date formats
- [ ] Add sort direction indicators
- [ ] Test with various date ranges"""
    },
    {
        "input": "Memory usage increases over time causing crash",
        "output": """## Bug Report: Memory Leak Causing Crashes

### Description
Application memory usage steadily increases over time, eventually causing crashes.

### Environment
- **Platform**: Web/Mobile Application
- **Component**: Memory Management

### Steps to Reproduce
1. Open application
2. Use normally for extended period
3. Monitor memory usage
4. Observe increasing consumption until crash

### Expected Behavior
Memory usage should remain stable during normal use.

### Actual Behavior
Memory grows unbounded until application crashes.

### Error Logs
```
Out of memory error
```

### Proposed Tasks
- [ ] Profile application for memory leaks
- [ ] Fix object disposal and cleanup
- [ ] Remove event listener leaks
- [ ] Implement memory monitoring
- [ ] Add automatic garbage collection hints"""
    },
    {
        "input": "Filtering combined with pagination shows wrong results",
        "output": """## Bug Report: Filter and Pagination Mismatch

### Description
When filters are applied, pagination shows incorrect results or wrong page count.

### Environment
- **Platform**: Web Application
- **Component**: Data Filtering/Pagination

### Steps to Reproduce
1. Apply filter to reduce results
2. Navigate to next page
3. Observe unfiltered results or incorrect count

### Expected Behavior
Pagination should respect active filters.

### Actual Behavior
Pagination ignores filters, showing incorrect data.

### Error Logs
```
Filter parameters not passed to pagination query
```

### Proposed Tasks
- [ ] Pass filter params to pagination API
- [ ] Reset to page 1 when filters change
- [ ] Update total count based on filter
- [ ] Sync filter state with URL params
- [ ] Add clear filters option"""
    },
    {
        "input": "Tooltip gets cut off at screen edge",
        "output": """## Bug Report: Tooltip Cut Off at Screen Edge

### Description
Tooltips are cut off when they appear near the edge of the screen.

### Environment
- **Platform**: Web Application
- **Component**: UI/Tooltips

### Steps to Reproduce
1. Hover over element near screen edge
2. Observe tooltip partially hidden

### Expected Behavior
Tooltips should reposition to stay fully visible.

### Actual Behavior
Tooltips extend beyond visible viewport.

### Error Logs
```
// CSS positioning issue
```

### Proposed Tasks
- [ ] Implement viewport boundary detection
- [ ] Add smart tooltip positioning
- [ ] Handle all screen edge cases
- [ ] Add arrow pointer adjustment
- [ ] Test on various screen sizes"""
    },
    {
        "input": "Share link generates 404 error",
        "output": """## Bug Report: Share Links Return 404

### Description
Share links generated by the application return 404 Not Found errors.

### Environment
- **Platform**: Web Application
- **Component**: Share/Link Generation

### Steps to Reproduce
1. Generate share link for content
2. Open link in new browser/incognito
3. See 404 error page

### Expected Behavior
Share links should load the shared content.

### Actual Behavior
All share links return 404 errors.

### Error Logs
```
Share route not registered in router
```

### Proposed Tasks
- [ ] Register share route handler
- [ ] Verify share link format
- [ ] Handle expired share links gracefully
- [ ] Add share link analytics
- [ ] Test share link generation and access"""
    },
    {
        "input": "Comments thread not loading replies",
        "output": """## Bug Report: Comment Replies Not Loading

### Description
Replies to comments are not loading, showing only parent comments.

### Environment
- **Platform**: Web Application
- **Component**: Comments/Discussion

### Steps to Reproduce
1. Navigate to post with comments
2. View comment with replies indicator
3. Click to expand replies
4. Replies do not load

### Expected Behavior
Comment replies should load when expanded.

### Actual Behavior
Reply section remains empty or shows loading indefinitely.

### Error Logs
```
Nested comments API endpoint returning empty array
```

### Proposed Tasks
- [ ] Debug replies API endpoint
- [ ] Fix nested comment query
- [ ] Implement lazy loading for replies
- [ ] Add reply count accuracy
- [ ] Handle deeply nested threads"""
    },
    {
        "input": "Multi-select dropdown only allows single selection",
        "output": """## Bug Report: Multi-Select Limited to Single Selection

### Description
Multi-select dropdown component only allows selecting one option at a time.

### Environment
- **Platform**: Web Application
- **Component**: Form/Select Components

### Steps to Reproduce
1. Click on multi-select dropdown
2. Select first option
3. Try to select second option
4. First selection is replaced

### Expected Behavior
Multiple options should be selectable simultaneously.

### Actual Behavior
Only one option can be selected at a time.

### Error Logs
```
// Component configuration issue
```

### Proposed Tasks
- [ ] Enable multiple selection mode
- [ ] Show selected items as chips/tags
- [ ] Add select all / clear all options
- [ ] Fix form value array handling
- [ ] Test keyboard multi-select"""
    },
    {
        "input": "Video playback stutters with buffering",
        "output": """## Bug Report: Video Playback Stuttering

### Description
Video playback frequently stutters and shows buffering, even on fast connections.

### Environment
- **Platform**: Web/Mobile Application
- **Component**: Video Player

### Steps to Reproduce
1. Open video for playback
2. Watch video playback
3. Observe frequent pauses for buffering

### Expected Behavior
Video should play smoothly with adequate buffer ahead.

### Actual Behavior
Constant stuttering and buffering interruptions.

### Error Logs
```
Buffer underrun, network latency spikes
```

### Proposed Tasks
- [ ] Implement adaptive bitrate streaming
- [ ] Increase buffer size
- [ ] Add preload hints
- [ ] Monitor and optimize CDN delivery
- [ ] Show buffer progress indicator"""
    },
    {
        "input": "Geographic location permission never requested",
        "output": """## Bug Report: Location Permission Not Requested

### Description
Features requiring location never prompt for permission, failing silently.

### Environment
- **Platform**: Web/Mobile Application
- **Component**: Geolocation

### Steps to Reproduce
1. Navigate to location-based feature
2. Feature fails or uses default location
3. No permission prompt appears

### Expected Behavior
User should be prompted for location permission.

### Actual Behavior
Permission is never requested, feature doesn't work.

### Error Logs
```
Geolocation permission not in required state
```

### Proposed Tasks
- [ ] Implement proper permission request flow
- [ ] Handle permission denial gracefully
- [ ] Add location permission explanation
- [ ] Provide manual location entry fallback
- [ ] Test on all platforms"""
    },
    {
        "input": "Undo function not working after save",
        "output": """## Bug Report: Undo Not Working Post-Save

### Description
The undo function stops working after saving changes, losing undo history.

### Environment
- **Platform**: Web Application
- **Component**: Edit/Undo System

### Steps to Reproduce
1. Make edits to content
2. Save changes
3. Try to undo
4. Undo button disabled or does nothing

### Expected Behavior
Undo history should persist or clearly indicate save boundary.

### Actual Behavior
Undo history is cleared on save without warning.

### Error Logs
```
Undo stack cleared on save operation
```

### Proposed Tasks
- [ ] Preserve undo stack across saves
- [ ] Or notify user history will be cleared
- [ ] Implement version history as alternative
- [ ] Add redo functionality
- [ ] Test undo/redo thoroughly"""
    },
]


def generate_training_data() -> List[Dict[str, str]]:
    """
    Generate synthetic training data for the LazyMode model.
    
    Returns:
        List of dictionaries with 'input' and 'output' keys.
    """
    all_templates = ISSUE_TEMPLATES + ADDITIONAL_TEMPLATES + MORE_TEMPLATES
    return all_templates


def save_dataset(data: List[Dict[str, str]], filepath: str) -> None:
    """
    Save the training dataset to a JSON file.
    
    Args:
        data: List of training examples.
        filepath: Path to save the JSON file.
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_dataset(filepath: str) -> List[Dict[str, str]]:
    """
    Load a training dataset from a JSON file.
    
    Args:
        filepath: Path to the JSON file.
        
    Returns:
        List of training examples.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_training_pairs(data: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    """
    Convert dataset into training pairs.
    
    Args:
        data: List of training examples.
        
    Returns:
        List of (input, output) tuples.
    """
    return [(item["input"], item["output"]) for item in data]


if __name__ == "__main__":
    # Generate and save training data
    training_data = generate_training_data()
    print(f"Generated {len(training_data)} training examples")
    
    # Save to file
    save_dataset(training_data, "data/training_data.json")
    print("Training data saved to data/training_data.json")
